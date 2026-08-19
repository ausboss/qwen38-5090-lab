#!/usr/bin/env python3
"""Inspect GGUF files for the things that decide whether one is worth serving here.

Reports: architecture, block count, whether a NextN/MTP draft head is present
(the single biggest speed lever on this box), quant mix, and the predicted
decode speed from the measured bandwidth model.

    bin/gguf-inspect.py <file.gguf> [more.gguf ...]
"""
import struct
import sys
from pathlib import Path

# GGUF value type ids
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FIXED = {U8: ('<B', 1), I8: ('<b', 1), U16: ('<H', 2), I16: ('<h', 2),
          U32: ('<I', 4), I32: ('<i', 4), F32: ('<f', 4), BOOL: ('<?', 1),
          U64: ('<Q', 8), I64: ('<q', 8), F64: ('<d', 8)}

# ggml tensor type id -> (name, bits per weight) for the ones we might meet
GGML = {0: ('F32', 32), 1: ('F16', 16), 2: ('Q4_0', 4.5), 3: ('Q4_1', 5),
        6: ('Q5_0', 5.5), 7: ('Q5_1', 6), 8: ('Q8_0', 8.5), 9: ('Q8_1', 9),
        10: ('Q2_K', 2.6), 11: ('Q3_K', 3.4), 12: ('Q4_K', 4.5), 13: ('Q5_K', 5.5),
        14: ('Q6_K', 6.6), 15: ('Q8_K', 8.5), 16: ('IQ2_XXS', 2.1),
        17: ('IQ2_XS', 2.3), 18: ('IQ3_XXS', 3.1), 19: ('IQ1_S', 1.6),
        20: ('IQ4_NL', 4.5), 21: ('IQ3_S', 3.4), 22: ('IQ2_S', 2.5),
        23: ('IQ4_XS', 4.25), 24: ('I8', 8), 25: ('I16', 16), 26: ('I32', 32),
        28: ('BF16', 16), 30: ('MXFP4', 4.25)}


def _read(f, fmt, n):
    return struct.unpack(fmt, f.read(n))[0]


def _rd_str(f):
    return f.read(_read(f, '<Q', 8)).decode('utf-8', 'replace')


def _rd_val(f, t):
    if t in _FIXED:
        fmt, n = _FIXED[t]
        return _read(f, fmt, n)
    if t == STR:
        return _rd_str(f)
    if t == ARR:
        et = _read(f, '<I', 4)
        n = _read(f, '<Q', 8)
        if et == STR:
            return [_rd_str(f) for _ in range(n)]
        if et in _FIXED:
            fmt, sz = _FIXED[et]
            return [_read(f, fmt, sz) for _ in range(n)]
        raise ValueError(f'array of type {et}')
    raise ValueError(f'value type {t}')


def inspect(path):
    p = Path(path)
    with open(p, 'rb') as f:
        assert f.read(4) == b'GGUF', 'not a GGUF file'
        _ver = _read(f, '<I', 4)
        n_tensors = _read(f, '<Q', 8)
        n_kv = _read(f, '<Q', 8)
        kv = {}
        for _ in range(n_kv):
            k = _rd_str(f)
            kv[k] = _rd_val(f, _read(f, '<I', 4))
        tensors = []
        for _ in range(n_tensors):
            name = _rd_str(f)
            nd = _read(f, '<I', 4)
            dims = [_read(f, '<Q', 8) for _ in range(nd)]
            ttype = _read(f, '<I', 4)
            _off = _read(f, '<Q', 8)
            n = 1
            for d in dims:
                n *= d
            tensors.append((name, ttype, n))

    arch = kv.get('general.architecture', '?')
    blocks = kv.get(f'{arch}.block_count')
    nextn = kv.get(f'{arch}.nextn_predict_layers')
    ctx = kv.get(f'{arch}.context_length')

    # The MTP/draft head is an EXTRA block past the transformer layers, declared
    # by nextn_predict_layers. Do NOT just look at the last block — in a build
    # with the head stripped, the last block is an ordinary transformer layer and
    # naive detection reports a false positive.
    has_mtp = bool(nextn) and nextn > 0
    head_blk = (blocks - 1) if (blocks and has_mtp) else None
    head = ([t for t in tensors if t[0].startswith(f'blk.{head_blk}.')]
            if head_blk is not None else [])
    head_bytes = sum(n * GGML.get(tt, ('?', 16))[1] / 8 for _, tt, n in head)

    total = p.stat().st_size
    # Streamed per decode step = everything except token_embd (a row gather)
    # and the draft head (only read when speculating).
    embd = sum(n * GGML.get(tt, ('?', 16))[1] / 8
               for nm, tt, n in tensors if nm.startswith('token_embd'))
    streamed = total - embd - head_bytes

    mix = {}
    for _, tt, n in tensors:
        mix[GGML.get(tt, (f'?{tt}', 16))[0]] = mix.get(GGML.get(tt, (f'?{tt}', 16))[0], 0) + n

    print(f'\n{p.name}')
    print(f'  size            {total/1e9:6.2f} GB   arch={arch} blocks={blocks} ctx={ctx}')
    print(f'  nextn layers    {nextn}')
    if head:
        print(f'  MTP head        PRESENT — blk.{head_blk}, {len(head)} tensors, '
              f'{head_bytes/1e9:.3f} GB  ->  --spec-type draft-mtp works')
    else:
        print('  MTP head        *** MISSING *** — no speculation, ~half the decode speed')
    top = sorted(mix.items(), key=lambda x: -x[1])[:4]
    print(f'  quant mix       {", ".join(f"{k}" for k, _ in top)}')
    print(f'  streamed/token  {streamed/1e9:6.2f} GB')
    print(f'  predicted       ~{1193/(streamed/1e9):5.0f} tok/s @1K   '
          f'(~{1193/(streamed/1e9)*0.85:.0f} @100K)')
    return {'name': p.name, 'gb': total/1e9, 'mtp': bool(head),
            'tok_s': 1193/(streamed/1e9)}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for a in sys.argv[1:]:
        try:
            inspect(a)
        except Exception as e:  # noqa: BLE001
            print(f'\n{Path(a).name}\n  FAILED: {e!r}')
