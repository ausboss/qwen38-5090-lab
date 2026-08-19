---
name: qwen38-local
description: Run, add, tune, and troubleshoot local LLMs on this machine's RTX 5090 (32GB) — the Qwen3.8-27B lab at ~/Documents/sglang-qwen3.8-lab, serving llama.cpp behind llama-swap (fast path, ~153 tok/s at 131K) and SGLang (long-prefill path, 184K). Use whenever the user wants to start/stop their local model, switch or download a checkpoint and serve it, run inference without a cloud key, work out whether a model fits in 32GB VRAM, change context length / thinking depth / sampling, reach the endpoint over LAN or Tailscale, register it in a client's model config, connect Open WebUI or ComfyUI, or debug a server that won't boot, is slow, OOMs, or gave less context than requested. Also for local inference speed, quantization choices (NVFP4/FP8/GGUF/Q3/Q4), speculative decoding, or long-context behaviour on this box — even when the user names no tool, and even when they say "SGLang" (the fast path is actually llama.cpp).
---

# Qwen3.8-27B, served locally (RTX 5090)

This machine serves local models from a lab directory. The user drives it with
one command, `qwen`, plus Open WebUI. Your job is usually one of: start/stop it,
put a different model behind it, change a setting, or work out why it's
misbehaving.

Two things make this box different from generic advice, and ignoring either
produces wrong answers:

1. **Decode speed is set by memory bandwidth, not compute.**
   `tok/s ≈ effective_BW / streamed_GB` (1,141 GB/s SGLang, 1,193 llama.cpp;
   streamed = body + `lm_head` only). No flag makes decoding faster — fewer
   streamed bytes does.
2. **On hybrid GDN models (Qwen3.8), context is cheap and concurrency is
   expensive.** That's backwards from a normal transformer, and it's why this box
   can serve 180K tokens on 32 GB.

If a **rig-info** skill is available, defer to it for hardware, CUDA, and
ComfyUI VRAM contention rather than re-deriving them. If an **lmstudio-local**
skill is available, note that LM Studio on `localhost:1234` is a separate,
independent server — both can be registered in a client at once, but only one
should hold the GPU.

## Layout

```
$LAB = ~/Documents/sglang-qwen3.8-lab
├── bin/qwen              # the user-facing command (symlinked to ~/.local/bin/qwen)
├── bin/serve.sh          # SGLang configs; one case per config
├── bin/net-setup.sh      # network exposure + API key rotation
├── bin/firewall-setup.sh # optional ufw layer (dry-run by default)
├── bin/backup-logs.sh    # archive journald logs for both services
├── bin/labauth.py        # shared bearer-token helper for the Python tools
├── bin/bench.py          # decode/prefill benchmark (cold vs warm prefix cache)
├── bin/needle.py         # long-context retrieval check
├── bin/smoke.py          # chat + multi-turn + image check
├── bin/gguf-inspect.py   # MTP head present? predicted tok/s
├── bin/webui-configure.py / webui-typecheck.py / webui-presets.sh
├── configs/llama-swap.yaml   # every llama.cpp build and its flags
├── configs/secrets/      # API key (gitignored)
├── skills/qwen38-local/  # THIS SKILL — the single source of truth
├── models/  logs/  results/
├── README.md  USAGE.md  REPORT.md  NOTES.md
```

**This skill lives in the repo** and is symlinked into each agent's skills
directory. Edit it at `$LAB/skills/qwen38-local/` — never edit through the
symlink path as if it were a separate copy, and never fork it per agent. It was
duplicated once and the copies silently drifted.

## Endpoints

| | |
|---|---|
| llama.cpp via llama-swap | `http://127.0.0.1:30001/v1` — **the default** |
| SGLang | `http://127.0.0.1:30000/v1` — only when `qwen long` is running |
| Open WebUI | `http://127.0.0.1:8080` |

The llama-swap endpoint also answers on the LAN IP, the Tailscale IP, and the
MagicDNS hostname. `qwen net` prints the current set.

**The endpoint requires a bearer token.** It listens on `0.0.0.0`, so requests
without one get **401**:

```bash
qwen key                                                    # print it
curl -H "Authorization: Bearer $(qwen key)" http://127.0.0.1:30001/v1/models
```

The key is in `configs/secrets/api-key.txt` (gitignored). Python tools in `bin/`
read it via `bin/labauth.py` — use that rather than hardcoding. SGLang on 30000
stays loopback-only with no auth.

## The builds, and when to use each

Despite what the user may call it, the fast path is **llama.cpp**, not SGLang.
All are vision-capable; all measured on this machine.

All llama.cpp builds sit behind llama-swap on one endpoint. `GET /v1/models`
lists every id whether or not it is loaded, and requesting an unloaded one swaps
to it in ~6s. **Never assume a model must be "started" first — just ask for it.**

| model id / command | runtime / build | decode @1K | context | note |
|---|---|---|---|---|
| `qwen38-fast` | llama.cpp UD-Q3_K_XL + MTP | **153 tok/s** | 131K | the default |
| `qwen38-uncfast` | llama.cpp Q4_K_S uncensored | 132 | 131K | leaves ~7GB for ComfyUI |
| `qwen38-uncensored` | llama.cpp Q6_K uncensored | 116 | 98K | best fidelity, no ComfyUI room |
| `qwen38-brief` | same as fast, `--reasoning-budget 128` | 153 | 65K | ~64% less output |
| `qwen long` | SGLang NVFP4 | 65 | 184K | separate server, 2x faster cold prefill |
| `qwen safe` | SGLang, lower mem-fraction | 65 | 131K | leaves GPU headroom |

Prefer `qwen long` only when repeatedly prefilling *fresh* ~100K documents.

## Everyday commands

```bash
qwen              # load the default fast build
qwen list         # every build the endpoint offers
qwen use <id>     # switch build, e.g. qwen use qwen38-uncfast
qwen long         # SGLang, biggest context
qwen ui           # start + Open WebUI chat page
qwen net          # all URLs this box answers on, plus the API key
qwen key          # print the API key alone
qwen status       # what's running
qwen gpu          # who is holding VRAM, per process
qwen test         # chat + multi-turn + image, end to end
```

Both `llama-swap` and `open-webui` run as **systemd user units with lingering
enabled**, so they start at boot without a login. Prefer
`systemctl --user restart llama-swap` over killing processes.

## Managing the GPU (this comes up constantly)

Two things hold VRAM and **both** must release before a big ComfyUI model fits:

```bash
qwen unload       # stops the LLM AND tells ComfyUI to drop its models.
                  # Web UI stays open. Frees ~26GB.
qwen              # bring the model back
qwen free         # ComfyUI only, LLM untouched
qwen stop         # model only — web UI keeps running
qwen down         # model + web UI
```

**ComfyUI caches models after every generation.** That is the usual reason a
model suddenly refuses to start: ComfyUI is silently sitting on ~18GB while
`nvidia-smi` shows plenty "free" to a careless reading. `qwen gpu` names the
process; `qwen free` fixes it.

## Thinking levels

Reasoning depth is a **model choice or a per-request kwarg** — not a restart.
`qwen think` takes no arguments; it prints the options.

```bash
qwen use qwen38-brief    # thinking capped at 128 tokens (~64% less output)
qwen use qwen38-fast     # normal reasoning
```

Per-request on any llama.cpp build, no reload:

```json
{ "chat_template_kwargs": {"reasoning_effort": "low"} }
{ "chat_template_kwargs": {"enable_thinking": false} }
```

Measured on one word problem, words inside `<think>`:

| setting | thinking | mechanism |
|---|---|---|
| `enable_thinking: false` | 0w | template branch |
| `--reasoning-budget 128` | 69w | **the only true hard cap** |
| `reasoning_effort: low` | 227w | template injects a "be brief" instruction |
| default | 271w | template default (`xhigh`) |
| `reasoning_effort: medium` | 435w | template injects **nothing** |

`medium` being the longest is not a mislabel — the Jinja has branches for
`xhigh` and `low` but no `elif` for medium, so it falls through with an empty
instruction and the model rambles. Name levels by measured behaviour, not by
what the template calls them.

Traps: a **top-level** `reasoning_effort` field and a per-request
`reasoning_budget` are both silently ignored — only the two kwarg forms above
work. On the SGLang path none of the levels work at all; SGLang inspects the
template at boot, concludes the model has no effort control, and drops the
value. Only `enable_thinking: false` does anything there.

## Working out if a model fits — do this before downloading

`references/memory-model.md` has the full arithmetic and how to inspect a
checkpoint's real dtype composition. The short version:

```
decode tok/s   ≈ effective_BW / streamed_GB        (×0.85 at 100K depth)
                 effective_BW: 1,141 SGLang · 1,193 llama.cpp
                 streamed = body + lm_head  (NOT embeddings/vision/draft head)
context tokens ≈ (28.5GB - weights - state_pool - 2.5GB) / 32KB    [hybrid GDN]
```

Two traps that cost hours here: **GiB reported as GB**, and counting the whole
checkpoint instead of only the streamed tensors. Each inflates predictions ~20%
and they compound. Don't estimate weight size from a model card — query the
HuggingFace API.

## Adding or swapping a model

Read `references/adding-a-model.md`. It routes GGUF work to
`configs/llama-swap.yaml` and safetensors work to `bin/serve.sh`, and covers the
size check, download with `hf`, the flags that matter, and verification.

For a GGUF, **always run `bin/gguf-inspect.py` first** — a missing MTP head costs
~40% decode speed and nothing warns you. Many abliterated/uncensored repacks
have it stripped.

Two habits worth keeping:

- **Verify, don't assume.** A server that boots is not a server that works. Run
  `qwen test`, and for long-context claims run `needle.py` — allocated KV cache
  tells you nothing about whether the model can attend to it.
- **Record the result in `REPORT.md`:** measured weight size, tok/s, actual
  context, needle pass rate. Otherwise the next session re-tests the same
  checkpoint.

## When something breaks

`references/troubleshooting.md` maps symptoms to real causes. Check it before
theorising — several failures here have misleading errors (a `ninja` error
that's really `PATH`; an OOM that's really speculative decoding resetting a
concurrency cap; a context length silently ignored; a 401 that is correct
behaviour).

Highest-value habit: **read the boot log before changing flags.**

```bash
grep -E "Load weight end|Mamba Cache|KV Cache is allocated|Memory pool end" \
  $LAB/logs/<config>.log            # SGLang
journalctl --user -u llama-swap -n 40   # llama.cpp — children log here too
```

Those lines say exactly where the VRAM went, which usually makes the fix obvious.

**Never `pkill -f <pattern>`** to stop these services. The pattern matches your
own command line and kills the shell running it (exit code 144). Kill by PID, or
use `systemctl --user stop`.

## Settings the user actually asks about

**Context length.** For llama.cpp, `-c` in the model's block in
`configs/llama-swap.yaml`. For SGLang, `--context-length` in `bin/serve.sh` —
where it is a *request*, not a guarantee; SGLang silently allocates less if
that's all that fits. Always confirm from the log.

**Sampling.** SGLang runs `sampling_defaults='model'`, so the checkpoint's
`generation_config.json` supplies defaults (Qwen3.8: temperature 1.0, top_k 20,
top_p 0.95). Override per-request, or per-model in Open WebUI.

**Connecting other tools.** Anything OpenAI-compatible: base URL
`http://127.0.0.1:30001/v1`, model = any id from `qwen list`, API key from
`qwen key`. This is how ComfyUI and scripts talk to it.

**Registering it in a client's model config** (e.g. pi's `~/.pi/agent/models.json`
under `providers`, same pattern as `lmstudio-local`):

```json
"qwen38": {
  "name": "Qwen3.8-27B (local, fast)",
  "baseUrl": "http://127.0.0.1:30001/v1",
  "api": "openai-completions",
  "apiKey": "<run: qwen key>",
  "models": [
    { "id": "qwen38-fast", "name": "Qwen3.8-27B 131K (llama.cpp)",
      "contextWindow": 131072, "maxTokens": 8192,
      "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 } }
  ]
}
```

Keeping `contextWindow` honest matters: a client's compaction math is wrong if
the number exceeds what the server really has, and the server truncates silently
rather than erroring. For the SGLang path read the real number:

```bash
grep "KV Cache is allocated" $LAB/logs/vision.log | tail -1
```

Back up before editing (`cp models.json models.json.bak-$(date +%s)`) and append
to `providers` — don't rewrite the object.

## The one non-obvious lesson

Speculative decoding's value is decided entirely by **the drafter's size**, not
by the technique:

| drafter | size | outcome |
|---|---|---|
| SGLang in-checkpoint MTP head | 5.53 GB bf16 | 64.8 → **50.3** tok/s, context 132K → 13K |
| llama.cpp GGUF MTP head (`blk.64`) | **0.21 GB quantized** | 85 → **153** tok/s, context unchanged |

Same model, same technique, opposite result. Four SGLang variants were tested
(including the official cookbook EAGLE recipe on two backends) and all lost —
even at 98% draft acceptance, because acceptance can't pay for a drafter costing
a quarter of the target model to read every step.

So when considering speculation: check the draft head's size and precision
first, and always benchmark against the no-speculation baseline. A GGUF often
ships a quantized MTP head where the safetensors build of the same model does
not — which can by itself be the reason to prefer llama.cpp for a given model.

## Constraints worth stating plainly to the user

- **ComfyUI and the big builds can't both be resident.** ComfyUI wants ~12 GB.
  `qwen38-fast` and `qwen38-uncfast` leave room; `qwen38-uncensored` (22.4 GB)
  does not.
- **A cold 100K prompt takes 25-49 s** before the first token; the same prefix
  again takes <1 s. Paste a long document once and ask many questions — don't
  re-send it each turn.
- **The fast path uses a Q3 quant.** It passed needle 9/9 at 100K and 8/8 on a
  reasoning probe, but that's a narrow check. If quality looks off on real work,
  `qwen38-uncfast` (Q4_K_S) or `qwen long` (full-precision NVFP4) are the
  fallbacks.
- **Context varies a few percent between boots** because `--mem-fraction-static`
  is a fraction of *total* VRAM and other apps move. Not a bug.

## Judgement calls

When the user wants "faster", establish which they mean — the fixes are
unrelated:

- **Faster first token on long prompts** → prefix caching, not decode. Keep the
  conversation prefix stable.
- **Faster token streaming** → smaller checkpoint, or a cheaper drafter. Nothing
  else moves it.
- **Snappier short replies** → `qwen use qwen38-brief`, or a terse system prompt.
