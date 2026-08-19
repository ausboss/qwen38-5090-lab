#!/usr/bin/env python3
"""Decode/prefill benchmark for Qwen3.8-27B on SGLang.

Measures what actually matters for an interactive agent: time-to-first-token and
sustained decode rate, at realistic context depths.

Streaming is required -- it is the only way to separate prefill (TTFT) from
decode. Output length is pinned with ignore_eos + min_tokens so every sample
decodes exactly the same number of tokens; otherwise a short generation inflates
tok/s and the comparison between configs is meaningless.

Usage:
  bin/bench.py --depths 1024,32768,102400 --concurrency 1 --gen 256 --label ctx128
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import labauth

LAB = Path(__file__).resolve().parent.parent
MODEL_DIR = LAB / "models" / "Qwen3.8-27B-NVFP4"


def build_prompt(tokenizer, target_tokens: int) -> str:
    """Return text whose tokenized length is very close to target_tokens.

    Uses varied prose rather than a repeated token: a degenerate repeated string
    is unrealistically friendly to prefix caching and to the GDN recurrent state.
    """
    filler = (
        "The maintenance log for reactor subsection {i} records a nominal "
        "coolant pressure of {p} kilopascals, with the secondary loop holding "
        "steady through the {i}th inspection cycle. Technician notes indicate "
        "no anomalies in the flow regulator assembly. "
    )
    chunks, i = [], 0
    while True:
        chunks.append(filler.format(i=i, p=1000 + (i * 7) % 900))
        i += 1
        if i % 16 == 0:  # only re-tokenize periodically; it is not cheap
            n = len(tokenizer.encode("".join(chunks)))
            if n >= target_tokens:
                break
        if i > 200000:
            break
    text = "".join(chunks)
    # Trim back to the exact target.
    ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(ids)


def one_request(base_url, model, prompt, gen_tokens, timeout=1800, nonce=None):
    """Fire one streaming completion. Returns (ttft, decode_s, n_tokens) or raises.

    `nonce` is prepended to the prompt to defeat the radix prefix cache. Without
    it, every repeat after the first is a cache hit and TTFT stops measuring
    prefill entirely -- the first version of this script reported 266,000 tok/s
    "prefill" at 100K depth, which is a cache hit, not compute.
    """
    if nonce is not None:
        prompt = f"[session {nonce}]\n{prompt}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "min_tokens": gen_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        # Without this the model re-renders <think> for prior turns and wrecks
        # prefix caching. Harmless single-turn; kept so the shape matches real use.
        "chat_template_kwargs": {"preserve_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers=labauth.headers(),
    )
    t0 = time.perf_counter()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            # Reasoning models stream <think> via reasoning_content; both are
            # decoded tokens and both count toward decode throughput.
            if delta.get("content") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    t_end = time.perf_counter()
    if ttft is None:
        raise RuntimeError("no tokens streamed")
    return ttft, t_end - t0 - ttft, n


def run_case(base_url, model, prompt, depth, gen, conc, repeats, cold=True):
    """cold=True gives every request a unique prefix so prefill is actually
    measured. cold=False reuses one prompt, which is the realistic multi-turn
    agent path where the prefix cache is warm."""
    results = []
    errors = []
    counter = itertools.count()

    def worker(nonce):
        try:
            results.append(one_request(base_url, model, prompt, gen, nonce=nonce))
        except Exception as e:  # noqa: BLE001 - surfaced in the report
            errors.append(repr(e))

    for rep in range(repeats):
        threads = []
        for c in range(conc):
            # Unique per (rep, stream) when cold; fixed when warm.
            nonce = f"{depth}-{next(counter)}" if cold else f"{depth}-warm"
            threads.append(threading.Thread(target=worker, args=(nonce,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    if not results:
        return {"depth": depth, "concurrency": conc, "cold": cold, "error": errors[:3]}

    ttfts = [r[0] for r in results]
    # Per-stream decode rate -- what a single user perceives.
    per_stream = [(r[2] - 1) / r[1] for r in results if r[1] > 0]
    total_tokens = sum(r[2] for r in results)
    return {
        "depth": depth,
        "concurrency": conc,
        "cold": cold,
        "gen_tokens": gen,
        "samples": len(results),
        "ttft_s": round(statistics.median(ttfts), 3),
        "decode_tps_per_stream": round(statistics.median(per_stream), 1),
        "decode_tps_aggregate": round(sum(per_stream), 1),
        "prefill_tps": round(depth / statistics.median(ttfts), 0) if depth else None,
        "tokens_returned": total_tokens,
        "errors": errors[:3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--depths", default="1024,32768,102400")
    ap.add_argument("--concurrency", default="1")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    depths = [int(x) for x in args.depths.split(",") if x]
    concs = [int(x) for x in args.concurrency.split(",") if x]

    print(f"[{args.label}] building prompts...", flush=True)
    prompts = {d: build_prompt(tok, d) for d in depths}
    for d, p in prompts.items():
        print(f"  depth {d:>7} -> {len(tok.encode(p)):>7} actual tokens", flush=True)

    if args.warmup:
        print("warmup...", flush=True)
        try:
            one_request(args.base_url, args.model, prompts[depths[0]], 32)
        except Exception as e:  # noqa: BLE001
            print(f"  warmup failed: {e!r}", flush=True)

    out = {"label": args.label, "gen_tokens": args.gen, "cases": []}
    for d in depths:
        for c in concs:
            for cold in (True, False):
                tag = "cold" if cold else "warm"
                print(f"running depth={d} conc={c} {tag} ...", end=" ", flush=True)
                r = run_case(args.base_url, args.model, prompts[d], d, args.gen, c,
                             args.repeats, cold=cold)
                out["cases"].append(r)
                if "error" in r:
                    print(f"FAILED {r['error']}", flush=True)
                else:
                    print(
                        f"decode {r['decode_tps_per_stream']} tok/s/stream, "
                        f"TTFT {r['ttft_s']}s, prefill {r['prefill_tps']:.0f} tok/s",
                        flush=True,
                    )

    path = Path(args.out) if args.out else LAB / "results" / f"bench_{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
