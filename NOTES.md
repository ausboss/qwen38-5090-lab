# Qwen3.8-27B on RTX 5090 — verified memory model

Everything here was measured on this machine, not taken from a doc. Boot logs in
`logs/`.

## Hardware / model

| | |
|---|---|
| GPU | RTX 5090, 32,607 MiB total (Blackwell, sm_120) |
| Held by desktop + sunshine + Claude Desktop | ~2.6 GB (permanent, don't plan around reclaiming it) |
| Model | `models/Qwen3.8-27B-NVFP4` (RadixArk), 21 GB on disk |
| Quant | `modelopt_mixed` / `MIXED_PRECISION` — **not** pure NVFP4 |
| Weights in VRAM | **20.14 GB** (measured, `Load weight end`) |
| Runtime | SGLang 0.5.17 + FlashInfer 0.6.15.post1, Python 3.12 |

> The widely-quoted "~16.5 GB NVFP4 weights" figure does **not** apply to this
> checkpoint. It loads at 20.14 GB because the linear-attn projections, MTP head,
> embeddings and `lm_head` stay at fp8/bf16. Planning against 16.5 GB is what
> makes configs mysteriously fail to fit.

## Architecture (from `config.json`)

64 layers = **16 full-attention** + **48 gated DeltaNet (linear)**, interleaved
3 GDN : 1 attention. `head_dim=256`, 4 KV heads, 262,144 native positions,
in-checkpoint MTP head (`mtp_num_hidden_layers: 1`).

Two separate memory pools follow from this, and they behave completely differently:

**KV cache — grows with context, only on 16 of 64 layers**

```
2 (K+V) x 4 kv_heads x 256 head_dim x 16 layers x 1 byte (fp8) = 32 KB / token
```

Measured: 132,857 tokens -> 4.06 GB. That is 32.0 KB/token. Formula confirmed.

**GDN recurrent state — fixed size, independent of context length**

```
48 layers x (48 v_heads x 128 x 128 ssm + conv) ~= 39.7M elements
  bf16 -> ~75 MB per slot ; 5 slots per in-flight request
```

Measured: 15 slots -> 1.16 GB, `max_running_requests` capped to 3.

**The consequence:** long context is *cheap* on this model and concurrency is
*expensive* — the opposite of a normal transformer. Doubling context costs
32 KB/token on 16 layers; adding one concurrent request costs ~375 MB flat.
For a single-user agent, trade concurrency away for context every time.

## The three flags that bought 41K tokens of context

Default settings gave only 91,867 KV tokens at `--context-length 131072` — SGLang
silently truncates instead of erroring. Fixes, in order of value:

| Flag | Reclaimed | Why |
|---|---|---|
| `--max-mamba-cache-size 15` | **1.25 GB** | Auto-sizing grabbed 32 slots (6 concurrent requests). Useless for one user. |
| `--mm-feature-transport cpu` | **1.00 GB** | This is a VLM; the `cuda_ipc` default reserves a flat 1024 MiB feature pool on GPU 0 even with zero images. |
| `--language-only` | (weights unchanged) | Skips the vision tower. Kept for correctness/startup, but it did **not** move the 20.14 GB figure. |

Result: **91,867 -> 132,857 KV tokens** at the same `--mem-fraction-static 0.90`.

## Gotchas that cost real time

- **`ninja` and `nvcc` must be on `PATH`.** Invoking `venv/bin/python` directly
  does not activate the venv, so FlashInfer's JIT dies with
  `FileNotFoundError: ninja`. `bin/serve.sh` exports both.
- **First boot JIT-compiles CUTLASS FP4 GEMMs for sm_120f** (minutes, 4 cores
  pegged). `FLASHINFER_WORKSPACE_BASE` is pinned to `$LAB/.flashinfer` so this
  is paid once, not per boot.
- **Never pass `--kv-cache-dtype` anything but `auto`.** The checkpoint ships fp8
  KV calibration scales; overriding discards them.
- **`--enable-linear-replayssm` (no `-spec`) is a batch>=64 optimization.** It is
  not useful here and is deliberately unused. The `-spec` variant is the one that
  matters for speculative decoding, and requires `--speculative-eagle-topk 1`.
- **`pkill -f 'sglang.launch_server'` kills your own shell.** Use the bracket
  trick: `pkill -f 'sglang.launch_serve[r]'`.
- **Multi-turn:** send `"chat_template_kwargs": {"preserve_thinking": false}` or
  the model re-renders empty `<think>` blocks for every prior turn and destroys
  prefix caching.

## Runtime choice: SGLang, not llama.cpp

llama.cpp does support this architecture (`qwen35` hybrid-GDN + `nextn` MTP head,
merged 2026-05) and GGUF quants exist. It is a viable fallback, but SGLang wins
here: native NVFP4 on Blackwell FP4 tensor cores, and a working in-checkpoint MTP
path. Reference point: llama.cpp users report ~42.9 tok/s on a 3090.

## Speculative decoding: MTP, not DSpark

DSpark needs a separate draft checkpoint (`RadixArk/Qwen3.8-27B-DSpark`) *and* an
offline-profiled SPS cost table — without the table "the budget degenerates to
verify-all (zero throughput gain by itself)". It is a batch-throughput technique;
the headline "206 tok/s on a 5090" is aggregate, not per-user. On single-stream
decode, MTP has far higher acceptance (0.23–0.52 vs 0.09–0.26) and a cheaper
4-token verify. For a single-user agent, MTP via the in-checkpoint head wins and
needs no extra download.
