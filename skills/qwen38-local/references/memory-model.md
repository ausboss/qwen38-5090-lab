# Sizing a model for this box — the arithmetic

Everything here was measured on the target machine, not taken from documentation.
Use it to predict whether a checkpoint fits *before* spending an hour downloading
it. The lab lives at `~/Documents/sglang-qwen3.8-lab`, called `$LAB` below.

## The budget

| | |
|---|---|
| RTX 5090 total | 32,607 MiB (~31.8 GiB) |
| Held by the desktop compositor + background apps | **~2.6-4 GB, varies** |
| Realistically available to the server | **~28-29.5 GB** |
| Effective bandwidth | **1,141 GB/s** SGLang · **1,193 GB/s** llama.cpp |

The compositor figure moves with what's on screen: 2.6 GB was measured during
this work, while the `rig-info` skill records 3-4 GB. Where a `rig-info` skill is
available, treat it as authoritative for hardware and budget for the pessimistic
end. This variance is also why the same config allocates a few percent different
context per boot.

## Rule 1 — decode speed is bandwidth, not compute

Generating one token streams the weights out of VRAM once. So:

```
decode tok/s  ~=  effective_BW  /  (STREAMED bytes per token)

  SGLang     effective_BW = 1,141 GB/s
  llama.cpp  effective_BW = 1,193 GB/s     (yes, llama.cpp is ~5% better here)
```

**"Streamed bytes" is not the checkpoint size.** Two things are easy to get
wrong, and getting them wrong throws the prediction off by 20%+:

- Use **decimal GB**, not GiB. A loader reporting "20.14 GB" is usually GiB
  (= 21.6 GB). Mixing the two silently inflates the answer.
- Count only what is **read every decode step**: the transformer body plus
  `lm_head`. Exclude `embed_tokens` (a row gather — one row per token, not the
  matrix), the vision tower (idle during text decode), and any MTP/draft head
  (only read when speculation is on).

Worked example, the RadixArk NVFP4 checkpoint:

```
body 16.893 + lm_head 0.715 = 17.608 GB streamed
1,141 / 17.608 = 64.8 tok/s          <- matches the measurement exactly
```

Total checkpoint is 21.9 GB, but 4.3 GB of that (embeddings 2.54, vision 0.92,
MTP 0.85) is never streamed. Predicting from 21.9 would have given 52 tok/s.

**Depth correction:** decode falls off with context. Measured here, apply
roughly **×0.85 at 100K** tokens.

The implication stands: **the main way to make decode faster is fewer streamed
bytes**, i.e. a smaller/better-quantized checkpoint. Both runtimes sit at 64-67%
of the 5090's 1,792 GB/s spec, which is normal — don't expect to tune that up.

## Rule 2 — for hybrid GDN models, context is cheap and concurrency is expensive

This is backwards from a normal transformer and is the key to large context here.

Qwen3.8-27B has 64 layers: **16 full-attention + 48 gated DeltaNet (linear)**.

- Only the 16 attention layers hold a KV cache:
  ```
  2 (K+V) x 4 kv_heads x 256 head_dim x 16 layers x 1 byte (fp8) = 32 KB / token
  ```
  Verified: 162,943 tokens -> 4.98 GB.

- The 48 GDN layers hold a **fixed-size recurrent state** that does *not* grow
  with context: ~75 MB per slot, and **5 slots per concurrent request**
  (~375 MB per request).

Consequences for a single-user setup:
- 100K of context costs 3.2 GB. Cheap.
- Allowing 6 concurrent requests costs 2.4 GB — for capacity you never use.
- **Always pin `--max-mamba-cache-size` low.** Left to auto-size it took 32 slots
  (2.41 GB). Ten slots (2 concurrent requests) is plenty and saves ~1.7 GB, worth
  ~53K tokens of context.

For a dense (non-hybrid) model, none of this applies — every layer caches KV, so
context is far more expensive and there's no mamba pool to tune.

## Rule 3 — putting it together

```
available (29.5 GB)
  - weights
  - mamba/GDN pool   (slots x 75 MB;  slots = 5 x max concurrent requests)
  - ~2.5 GB          (activations, CUDA graphs, fragmentation)
  = left for KV

context tokens = KV bytes / 32 KB
```

Worked example, current checkpoint:
```
29.5 - 20.14 - 0.80 - 2.5 = 6.06 GB  ->  6.06e9 / 32768 = ~185K tokens
```
Observed: 184,183 tokens. The model predicts reality closely.

## Rule 4 — SGLang silently truncates

Asking for `--context-length 204800` when only 160K fits produces **no warning**.
It quietly allocates what it can. Always confirm what you actually got:

```bash
grep "KV Cache is allocated" logs/<config>.log
# -> KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 162943, ...
```

Treat `--context-length` as a ceiling request, never as a guarantee.

## Rule 5 — expect a few percent of boot-to-boot variance

`--mem-fraction-static` is a fraction of *total* VRAM, so whatever else is on the
GPU when the server boots changes the outcome. The same config has produced
125,555 / 154,718 / 160,274 / 162,943 / 184,183 KV tokens on different boots.
Don't tune to the last token, and don't be alarmed by the drift.

## Reading the boot log

These four lines tell you everything about how memory was spent:

```
Load weight end. ... mem usage=20.14 GB          <- weights
Mamba Cache is allocated. max_mamba_cache_size: 10, ... ssm_state size: 0.77GB
KV Cache is allocated. ... #tokens: 162943        <- ACTUAL context
Memory pool end. avail mem=2.23 GB               <- headroom left
```

If `avail mem` at the end is under ~1 GB, the GPU is full enough that launching
another GPU app may fail. Drop `--mem-fraction-static` by 0.02 to get slack back.

## Estimating a checkpoint's VRAM before downloading

Weight files on disk ≈ VRAM used for weights. Get the real number from the
HuggingFace API rather than the model card, which is often optimistic:

```bash
curl -s "https://huggingface.co/api/models/<repo>/tree/main" \
  | python3 -c "import sys,json;print(sum(f.get('size',0) for f in json.load(sys.stdin))/2**30,'GiB')"
```

Then check what fraction is *genuinely* low-precision — a checkpoint advertised
as "NVFP4" is often mixed, with embeddings, `lm_head`, norms and any MTP head
left at bf16. Inspect `config.json` -> `quantization_config` for the
ignored/excluded layer list. To measure a local checkpoint exactly:

```python
from safetensors import safe_open
import glob, collections
tot = collections.Counter()
for f in glob.glob("<model dir>/*.safetensors"):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            sl = sf.get_slice(k); n = 1
            for d in sl.get_shape(): n *= d
            bpe = {"F32":4,"F16":2,"BF16":2,"F8_E4M3":1,"U8":1}.get(sl.get_dtype(), 2)
            tot[sl.get_dtype()] += n * bpe
for dt, b in tot.most_common():
    print(f"{dt:>10} {b/2**30:7.2f} GB")
```

Packed FP4 shows up as `U8` (two 4-bit values per byte).
