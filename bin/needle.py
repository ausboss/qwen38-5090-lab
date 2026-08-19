#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval check for Qwen3.8-27B at long context.

Throughput numbers are meaningless if the model cannot actually use the context
it is configured for. This plants a unique fact at several relative depths in a
long filler document and checks whether the model can recall it verbatim.

The needle is a random-looking 6-digit code tied to an unusual noun phrase, so a
correct answer cannot be guessed from the filler or from parametric knowledge.

Usage:
  bin/needle.py --depths 8192,32768,102400 --positions 0.1,0.5,0.9 --label ctx128
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
MODEL_DIR = LAB / "models" / "Qwen3.8-27B-NVFP4"

# Deterministic per (depth, position) so a rerun is comparable, but distinct
# across cells so the model cannot carry an answer between cells.
def make_code(depth: int, pos: float) -> str:
    h = (depth * 7919 + int(pos * 1000) * 104729) % 900000 + 100000
    return str(h)


NEEDLE_TMPL = (
    "IMPORTANT RECORD: The calibration key for the Vermillion Harbor tide gauge "
    "is {code}. Remember this exact number."
)

QUESTION = (
    "What is the calibration key for the Vermillion Harbor tide gauge? "
    "Reply with only the number, nothing else."
)


def build_haystack(tokenizer, target_tokens: int, needle: str, position: float) -> str:
    filler_unit = (
        "Quarterly logistics review notes: shipment {i} cleared customs without "
        "incident, container seals intact, manifest reconciled against the bill "
        "of lading. Warehouse {i} reported nominal humidity and no pallet damage. "
    )
    chunks, i = [], 0
    while True:
        chunks.append(filler_unit.format(i=i))
        i += 1
        if i % 32 == 0 and len(tokenizer.encode("".join(chunks))) >= target_tokens:
            break
        if i > 300000:
            break
    ids = tokenizer.encode("".join(chunks))[:target_tokens]
    insert_at = max(0, min(len(ids) - 1, int(len(ids) * position)))
    head = tokenizer.decode(ids[:insert_at])
    tail = tokenizer.decode(ids[insert_at:])
    return f"{head}\n\n{needle}\n\n{tail}"


def ask(base_url, model, prompt, timeout=1800):
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"{prompt}\n\n{QUESTION}"},
        ],
        "max_tokens": 2048,  # room for the <think> block before the answer
        "temperature": 0.0,
        "chat_template_kwargs": {"preserve_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    return msg.get("content") or "", time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--depths", default="8192,32768,102400")
    ap.add_argument("--positions", default="0.1,0.5,0.9")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    depths = [int(x) for x in args.depths.split(",") if x]
    positions = [float(x) for x in args.positions.split(",") if x]

    cells, passed, total = [], 0, 0
    for d in depths:
        for p in positions:
            code = make_code(d, p)
            hay = build_haystack(tok, d, NEEDLE_TMPL.format(code=code), p)
            actual_tokens = len(tok.encode(hay))
            print(f"depth={d:>7} pos={p:<4} ({actual_tokens} tok) ...", end=" ", flush=True)
            try:
                answer, elapsed = ask(args.base_url, args.model, hay)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR {e!r}", flush=True)
                cells.append({"depth": d, "position": p, "error": repr(e)})
                total += 1
                continue
            found = re.findall(r"\d{6}", answer)
            ok = code in found
            passed += ok
            total += 1
            print(f"{'PASS' if ok else 'FAIL'} ({elapsed:.1f}s) expected {code}, got {found[:3] or answer[:60]!r}", flush=True)
            cells.append({
                "depth": d, "position": p, "tokens": actual_tokens,
                "expected": code, "answer_codes": found[:3],
                "pass": ok, "latency_s": round(elapsed, 1),
            })

    out = {"label": args.label, "passed": passed, "total": total, "cells": cells}
    path = Path(args.out) if args.out else LAB / "results" / f"needle_{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n{passed}/{total} passed -> wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
