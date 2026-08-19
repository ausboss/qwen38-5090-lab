# Qwen3.8-27B as a fast local agent on one RTX 5090 — findings

**Goal:** run Qwen3.8-27B locally as an agent that is smart, fast, and has at
least 100K of usable context, on an RTX 5090 (32 GB).

**Result: 153 tok/s at short context, 80 tok/s at 100K, 131K context, 9/9 needle,
8/8 reasoning, vision working.** Both goals met — 2.4x the original speed with
more than the required context.

```bash
qwen
```

Serves an OpenAI-compatible endpoint at `http://127.0.0.1:30001/v1`, model name
`qwen38-fast`.

### How it got there (each step measured)

| config | @1K | @100K | context |
|---|---|---|---|
| SGLang NVFP4 (first working setup) | 64.8 | 55.2 | 184K |
| llama.cpp Q4_K_M | 75.4 | 52.4 | 131K |
| llama.cpp UD-Q3_K_XL | 85.2 | 56.7 | 131K |
| **llama.cpp UD-Q3_K_XL + MTP** | **153.0** | **79.8** | **131K** |

The last row is the one that matters, and it works for a specific reason:
**that GGUF ships a *quantized* MTP head (0.21 GB)**. The same speculation in
SGLang uses a bf16 head costing 5.53 GB, which is why it lost there four times
running. Same technique, 26x cheaper drafter, opposite outcome.

Quality checks on the Q3 quant (an aggressive one, so these matter):
needle **9/9** at 8K/32K/100K x 3 positions, and **8/8** on a reasoning probe
(arithmetic, syllogism, the bat-and-ball trap, sibling counting, code).

---

## 1. Measured results

All numbers from this machine. Raw data in `results/`, boot logs in `logs/`.
Decode is per-stream at concurrency 1 — what one user actually feels.

### Recommended config — `ctx200` (154,718 tokens)

| prompt depth | decode | cold TTFT | notes |
|---|---|---|---|
| 1K | **64.0 tok/s** | 0.11 s | |
| 100K | **54.7 tok/s** | 24.8 s | cold = nothing cached |

### `ctx128` (125,555 tokens) — measured in more detail

| depth | decode | cold TTFT | warm TTFT |
|---|---|---|---|
| 1K | 64.8 tok/s | 0.12 s | 0.10 s |
| 32K | 61.1 tok/s | 4.4 s | 0.18 s |
| 100K | 55.2 tok/s | 24.6 s | **0.54 s** |

The two configs decode identically (64.0 vs 64.8 at 1K; 54.7 vs 55.2 at 100K).
**+30K of context costs nothing in speed**, so take the bigger one.

### Long-context retrieval — passes cleanly

Unique 6-digit code planted in filler prose, retrieved verbatim:

| config | depths x positions | result |
|---|---|---|
| ctx128 | 8K / 32K / 100K x 10% / 50% / 90% | **9/9 pass** |
| ctx200 | 131,101 tokens x 10% / 50% / 90% | **3/3 pass** |

The 50% position is the classic "lost in the middle" failure mode and it passed
at every depth. The context is genuinely usable, not just allocated.

### Decode barely degrades with depth

64.8 -> 55.2 tok/s going from 1K to 100K: a **14% falloff across a 100x context
increase**. That is the hybrid architecture paying off — 48 of 64 layers hold a
fixed-size recurrent state and never touch a growing KV cache.

---

## 2. The thing that actually limits the experience

**It is not decode speed. It is cold prefill.**

At 100K, first-token latency is **24.6 s** cold versus **0.54 s** warm — a **46x
gap**. Decode differences of 55 vs 65 tok/s are noise next to that.

Two consequences for using this as an agent:

1. **Protect the prefix cache.** On every multi-turn request send:
   ```json
   "chat_template_kwargs": {"preserve_thinking": false}
   ```
   Without it the model re-renders empty `<think>` blocks for every prior turn,
   which changes the prefix, misses the cache, and costs you 24 seconds per turn.
2. **Long context is cheap to *hold*, expensive to *fill*.** Loading a 100K
   document once and then conversing over it is fast. Re-sending a different
   100K document every turn is not.

---

## 3. Decisions, with the evidence

### MTP speculative decoding — tested and rejected

This was the previous plan's centerpiece, expected to roughly double decode. It
does the opposite here:

| | context | decode @1K |
|---|---|---|
| no MTP | **132,857 tok** | **64.8 tok/s** |
| MTP | 12,985 tok | 50.3 tok/s |

**22% slower and 90% less context.** The cause is measurable in the boot log:

```
Load weight end. type=Qwen3_5ForCausalLMMTP ... mem usage=5.53 GB
```

Despite `mtp_num_hidden_layers: 1`, the draft head carries its own bf16
embeddings and `lm_head` over a 248,320-token vocab — 5.53 GB, roughly 170K
tokens' worth of KV cache. And because decode here is VRAM-bandwidth-bound,
re-reading those draft weights every draft step costs more than the accepted
tokens save.

Speculation itself worked fine (mean accept length **2.98**, accept rate
**0.66**) — the draft is accurate, it is just too expensive. `--enable-linear-
replayssm-spec` also did its job, holding verify buffers to 0.01 GB; it simply
cannot offset a 5.53 GB draft head.

*Bounded investigation:* spec parameters were not exhaustively swept, because MTP
is disqualified on the 100K context requirement at any tuning.

### MTP with a cheaper draft (1 step) — also tested, still loses

Follow-up test, since "we drafted too aggressively" was the obvious suspect:

| config | decode | accept length |
|---|---|---|
| no MTP | **64.8 tok/s** | — |
| MTP, 1 draft step | 56.5 tok/s | 1.97 / 2 (**98% accepted**) |
| MTP, 3 draft steps | 50.3 tok/s | 2.98 / 4 |

Fewer draft steps helps (50.3 -> 56.5) but never catches plain decoding, **even at
98% acceptance**. That rules out tuning as the explanation. The cause is that the
draft head is **bf16** while the target model is **NVFP4** — the "cheap helper"
runs on the slow path, off the FP4 tensor cores, and has to compute a full
248,320-row vocabulary projection on every guess. A checkpoint shipping a
*quantized* MTP head would likely flip this result; this one does not.

### DSpark — rejected without testing

Needs a separate draft checkpoint plus an offline-profiled SPS cost table
(without the table it "degenerates to verify-all, zero throughput gain"). It is a
*batch throughput* technique; the widely-quoted "206 tok/s on a 5090" is
aggregate, not per-user. On single-stream decode, published acceptance is
0.09–0.26 versus MTP's 0.23–0.52 — and MTP already lost here.

### SGLang over llama.cpp

llama.cpp genuinely supports this architecture now (`qwen35` hybrid-GDN plus the
`nextn` MTP head, merged 2026-05), so it is a real fallback. SGLang wins on
native NVFP4 execution on Blackwell FP4 tensor cores. Reference point: llama.cpp
users report ~42.9 tok/s on a 3090.

---

## 4. Why the config looks the way it does

The single most useful fact about this model:

> **Long context is cheap. Concurrency is expensive.** The opposite of a normal
> transformer.

64 layers = 16 full-attention + 48 gated DeltaNet. Only the 16 attention layers
cache KV, at a measured **32 KB/token**. The 48 GDN layers hold a *fixed-size*
recurrent state — ~75 MB per slot, 5 slots per in-flight request — that does not
grow with context at all.

So one extra concurrent request costs ~375 MB flat, while 12,000 extra tokens of
context cost the same. For a single-user agent, trade concurrency away for
context every time. That is what these flags do:

```
--max-mamba-cache-size 10     # not 32 (the auto-sized default) -> +1.7 GB
--mm-feature-transport cpu    # VLM default pins 1024 MiB on GPU for images
                              # you will never send -> +1.0 GB
--language-only               # skip the vision tower
--cuda-graph-bs-decode 1 2 4  # don't capture graphs for batch sizes the GDN
                              # pool makes unreachable
```

Default settings gave **91,867** tokens at `--context-length 131072`. The same
memory budget with these flags gives **154,718**. Nothing was traded away except
concurrency this workload never uses.

**Watch out:** SGLang **silently truncates** context to whatever fits. Asking for
131072 and getting 91867 produces no warning — you have to read
`KV Cache is allocated ... #tokens:` in the boot log. Never assume
`--context-length` is what you got.

---

## 5. Practical notes

- **`ctx200` runs the GPU nearly full — ~124 MiB free while serving.** It is
  stable, but there is no headroom: launching another GPU app while it runs may
  fail to allocate. If you want to keep working on the desktop alongside it, use
  `bin/serve.sh ctx128` instead (`--mem-fraction-static 0.90`, ~125K context,
  identical decode speed, ~2 GB of slack). The 30K of extra context is the only
  thing you give up.
- **Actual context varies ~125K–155K between boots** with identical flags.
  `--mem-fraction-static` is a fraction of *total* VRAM, so whatever else is on
  the GPU at boot (compositor, Claude Desktop) shifts the result. Don't tune to
  the last token.
- **First boot after an SGLang upgrade takes several minutes** — FlashInfer
  JIT-compiles CUTLASS FP4 GEMMs for sm_120f. The cache is pinned to
  `$LAB/.flashinfer`, so this is paid once. Later boots are ~60 s.
- **Thinking is always on** (26 reasoning tokens even for "reply PONG"). Budget
  `max_tokens` accordingly or short answers get truncated mid-`<think>`.
- **ComfyUI and the LLM do coexist** — that was later measured, not assumed.
  With the Q3_K_XL build and Z-Image Turbo fp8 (text encoder on CPU), peak was
  27.7 GB of 32.6 GB. The bigger builds are tighter; see `qwen gpu`.
- `qwen stop` / `status` / `unload` manage the server. `qwen unload` frees both
  the LLM *and* ComfyUI's cached models while leaving the web UI open.

### How to actually get faster than ~65 tok/s

> **Corrected 2026-08-19.** The earlier version of this section said
> `tok/s = 1305 / weight_GB`. That was wrong twice: it mislabelled GiB as GB,
> and it counted bytes that are never streamed during decode (`embed_tokens` is
> a row gather, the vision tower is idle, the MTP head is only read when
> speculation is on). The corrected model below predicts the measurements
> exactly; the old one was off by ~20%.

Decode speed is set by **memory bandwidth**, not compute — but only the bytes
actually streamed each step count:

```
streamed = transformer body + lm_head        (NOT embeddings/vision/MTP)

SGLang    17.608 GB streamed -> 64.8 tok/s  =>  1,141 GB/s effective
llama.cpp 15.822 GB streamed -> 75.4 tok/s  =>  1,193 GB/s effective
```

Both sit at 64-67% of the 5090's 1,792 GB/s spec. Note llama.cpp is ~5% *more*
bandwidth-efficient than SGLang here — the opposite of what was assumed earlier.
Apply roughly **×0.85 at 100K** depth.

There is no tuning left in that 73%. The only lever that matters is **fewer
bytes per token**, i.e. a smaller checkpoint. Measured composition of what we
have now, read straight from the safetensors:

| dtype | size | what it is |
|---|---|---|
| packed FP4 (U8) | 8.56 GB | the actually-4-bit weights |
| FP8_E4M3 | 7.79 GB | 8-bit — could be 4-bit |
| BF16 | 4.07 GB | 16-bit — embeddings, `lm_head`, norms, MTP head |
| **total** | **20.42 GB** | |

Only 42% of this checkpoint is genuinely 4-bit. Projected speeds at the same
73% bandwidth efficiency:

| checkpoint | decode |
|---|---|
| today (mixed, 20.1 GB) | 65 tok/s |
| FP8 portion also 4-bit (16.2 GB) | ~80 tok/s |
| **full NVFP4 (~13.5 GB)** | **~97 tok/s** |

**So ~100 tok/s needs a ~13.5 GB checkpoint, not a config change.** That would
also free ~6.6 GB for context, and might make MTP viable (a quantized draft head
would be cheap enough to pay for itself). This is the single highest-value thing
left to try.

For reference, 100,000 tok/s single-stream is not physically reachable — it would
need roughly 2 PB/s of memory bandwidth, about 1,000x this card.

### If you want more context than 155K

The ceiling is set by weights, and this checkpoint is heavier than advertised:
it loads at **20.14 GB**, not the widely-quoted ~16.5 GB, because it is
`MIXED_PRECISION` (fp8 for the linear-attn projections, embeddings and `lm_head`
in bf16) rather than pure NVFP4. A leaner checkpoint is the only big lever left —
`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` advertises 18.8 GB resident and
claims full 262K on a 5090. Untested here; it would be the next thing to try,
and ~1.4 GB of weights is worth roughly 44K tokens of context.

---

## 6. Files

```
bin/qwen              # the daily driver — every command lives here
bin/serve.sh          # SGLang configs (vision, safe, eagle, ctx*, mtp*)
bin/bench.py          # decode/prefill benchmark, cold vs warm prefix cache
bin/needle.py         # long-context retrieval check
bin/smoke.py          # chat + multi-turn + image check
bin/gguf-inspect.py   # does a GGUF have an MTP head? predicted tok/s
bin/webui-configure.py # apply the whole Open WebUI config
bin/webui-typecheck.py # catch wrong-shaped config values before they break the UI
bin/webui-presets.sh  # create the Open WebUI model presets
configs/              # the ComfyUI Z-Image workflow used for image generation
USAGE.md              # plain-English guide — start here
NOTES.md              # verified memory model + gotchas
results/              # raw benchmark + needle JSON
logs/                 # boot logs (memory breakdowns live here)
```

### Benchmark methodology note

An early version of `bench.py` reported 266,000 tok/s "prefill" at 100K depth.
That was the radix prefix cache serving a repeated prompt, not compute. The
script now prepends a unique nonce per request to force a true cold measurement,
and reports cold and warm separately. Use `--repeats 3` or higher — with an even
repeat count the median of `[miss, hit]` is a meaningless midpoint.
