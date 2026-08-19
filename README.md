# Qwen3.8-27B on one RTX 5090

Running Qwen3.8-27B locally at **153 tok/s with 131K context**, with vision,
image generation, and an uncensored variant — all on a single 32GB card that is
also driving a desktop.

This started as an attempt to find the best SGLang config and ended somewhere
else entirely. The measurements are the point; the numbers below were taken on
this machine, not copied from model cards.

```bash
qwen            # start it
qwen ui         # start it + the chat page
qwen status     # what's running
```

New here? Read **[USAGE.md](USAGE.md)** — it's the plain-English guide.

---

## What's running

| build | decode @1K | @100K | context | notes |
|---|---|---|---|---|
| `qwen` | **153 tok/s** | 80 | 131K | Q3_K_XL + MTP, the default |
| `qwen uncfast` | 132 | — | 131K | uncensored, leaves room for ComfyUI |
| `qwen uncensored` | 116 | — | 98K | uncensored, best fidelity |
| `qwen long` | 65 | 55 | 184K | SGLang NVFP4, 2× faster cold prefill |

All are vision-capable. All verified with `bin/needle.py` (long-context recall)
and a reasoning probe, not just "it booted".

## The three findings worth stealing

**1. Speculative decoding lives or dies on the drafter's size.**

| drafter | size | result |
|---|---|---|
| SGLang's in-checkpoint MTP head | 5.53 GB bf16 | 64.8 → **50.3** tok/s |
| the GGUF's MTP head | **0.21 GB quantized** | 85 → **153** tok/s |

Same model, same technique, opposite outcome. Four SGLang variants were tested —
including the official cookbook EAGLE recipe on two attention backends — and all
four lost, even at 98% draft acceptance. Acceptance cannot pay for a drafter that
costs a quarter of the target model to read every step.

**2. On a hybrid GDN model, long context is cheap and concurrency is expensive.**
Only 16 of Qwen3.8's 64 layers hold a KV cache (32 KB/token); the other 48 hold a
fixed-size recurrent state that doesn't grow with context but costs ~375 MB per
concurrent request. Trading concurrency away took the same memory budget from
91,867 to 184,183 tokens.

**3. Cold prefill, not decode, is what you actually feel.** At 100K it's ~25-49 s
for a fresh document versus ~0.5 s once cached. Paste once, ask many questions.

More detail — including the corrected bandwidth model and every config that
lost — is in **[REPORT.md](REPORT.md)**.

## Layout

```
bin/qwen                 the daily driver — all commands
bin/serve.sh             SGLang configs
bin/bench.py             decode/prefill benchmark (cold vs warm prefix cache)
bin/needle.py            long-context retrieval check
bin/smoke.py             chat + multi-turn + image check
bin/gguf-inspect.py      does a GGUF have an MTP head? predicted tok/s
bin/webui-configure.py   apply the whole Open WebUI config
bin/webui-typecheck.py   catch wrong-shaped config values before they break the UI
bin/webui-presets.sh     create the Open WebUI model presets
configs/                 ComfyUI Z-Image workflow used for image generation
results/                 raw benchmark + needle JSON for every config tested
USAGE.md                 plain-English guide
REPORT.md                measurements and reasoning
NOTES.md                 verified memory model + failure modes
```

`models/`, `venv/` and `logs/` are gitignored — they're tens of GB and machine-specific.

## Reproducing

Nothing here is a general-purpose tool; it's tuned to one machine (RTX 5090,
32GB, CUDA 13.1, Kubuntu). The parts that transfer are the method and the
numbers:

```bash
bin/gguf-inspect.py <any.gguf>     # MTP head present? predicted tok/s?
bin/bench.py  --depths 1024,102400 --repeats 3 --label mine
bin/needle.py --depths 8192,102400 --positions 0.1,0.5,0.9 --label mine
```

`bench.py` reports **cold** and **warm** separately — cold is true prefill, warm
is a prefix-cache hit. Comparing one against the other is the easiest way to
publish a wrong number; an early version of it reported 266,000 tok/s "prefill"
that was entirely cache hits.

## Credits

Model: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ·
GGUF quants: [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) ·
uncensored builds: [JonathanColetti](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF),
[mradermacher](https://huggingface.co/mradermacher/Qwen3.8-27B-Uncensored-Aggressive-GGUF) ·
runtimes: [llama.cpp](https://github.com/ggml-org/llama.cpp), [SGLang](https://github.com/sgl-project/sglang)
