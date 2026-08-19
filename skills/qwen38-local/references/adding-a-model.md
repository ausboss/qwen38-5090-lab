# Adding a new model

The workflow: decide whether it fits, get it, give it a config, boot it, verify
it, then record what you learned. Skipping the "decide whether it fits" step is
how you lose an hour to a download that was never going to work.

## Which path — read this first

There are two runtimes here and **llama.cpp is the fast one**, despite this
skill's name. Default to it.

| the model is | use | where the config goes |
|---|---|---|
| a **GGUF** | llama.cpp behind llama-swap | `configs/llama-swap.yaml` |
| safetensors (fp8/NVFP4/bf16) | SGLang | `bin/serve.sh` |

For a GGUF, skip to *Adding a GGUF build to llama-swap* below — you do not touch
`bin/serve.sh` and you do not restart anything by hand.

## Adding a GGUF build to llama-swap

Everything llama.cpp serves is a block in `configs/llama-swap.yaml`. Adding one
makes it appear in `GET /v1/models` permanently, so Open WebUI presets and
scripts can bind to the id whether or not it is currently loaded.

**Check the MTP head before committing to a GGUF.** This is the single biggest
speed factor and nothing warns you when it is missing:

```bash
bin/gguf-inspect.py /path/to/model.gguf
```

`blocks=65, nextn layers=1` means the draft head is present (~2x decode).
`blocks=64, nextn layers=None` means it was stripped — common in abliterated and
uncensored repacks, because re-saving through transformers drops the `mtp.*`
tensors while `config.json` still advertises them. A stripped build runs ~40%
slower and needs `--spec-type draft-mtp` removed from its `cmd`, or llama-server
will fail to start.

Then copy the nearest block:

```yaml
  qwen38-mynew:
    name: "Human readable name"
    description: "Shows up in the model picker."
    cmd: >
      llama-server ${common}
      -m ${lms}/org/repo/model.gguf
      --mmproj ${lab}/models/.../mmproj-BF16.gguf
      -c 131072
```

`${common}` carries the flags that are easy to get wrong: `-fa on` (**required**
for the quantized KV cache), `-ngl 999`, `--no-context-shift`, the q4_0 KV cache
types, and `--spec-type draft-mtp`. `${lab}` and `${lms}` are path macros.

Pick `-c` from the weight size — the KV cache has to fit in what's left of the
~28-29.5 GB budget. 22 GB of weights only leaves room for ~98K.

```bash
systemctl --user restart llama-swap
qwen list                       # the new id should appear
qwen use qwen38-mynew           # loads it; ~6s
qwen test                       # chat + multi-turn + image
```

If it fails to load, `journalctl --user -u llama-swap -n 40` has the
llama-server output — the children log to the same journal.

The rest of this file is the **SGLang** path, for safetensors checkpoints.

## Step 0 — will it fit, and will it be fast enough?

Do this before downloading anything. Get the real weight size from the HF API,
not the model card:

```bash
REPO=someorg/some-model
curl -s "https://huggingface.co/api/models/$REPO/tree/main" \
  | python3 -c "import sys,json;fs=json.load(sys.stdin);print(round(sum(f.get('size',0) for f in fs)/2**30,2),'GiB')"
```

Note that returns **GiB**, not GB — multiply by 1.074 before using it below.
Mixing the two is a 7% error that compounds with the next one.

Then apply the arithmetic in `memory-model.md`:

- **Speed:** `decode tok/s ≈ effective_BW / streamed_GB`, where `effective_BW` is
  1,193 GB/s on llama.cpp or 1,141 GB/s on SGLang, and `streamed_GB` counts only
  the transformer body plus `lm_head` — **not** embeddings, the vision tower, or
  a draft head. Apply ×0.85 at 100K depth.
- **Fit:** total weights must leave room for KV + state + ~2.5 GB overhead inside
  ~28-29.5 GB.

Rules of thumb for this 32 GB card (total checkpoint size, not streamed):
- **≤16 GB** — comfortable, lots of context, ~85+ tok/s
- **16–22 GB** — works, context gets tight above 20 GB
- **22–26 GB** — only with small context; check carefully
- **>26 GB** — won't work without offload, which destroys decode speed

Also confirm SGLang actually supports the architecture. Check that the
`architectures` field in `config.json` maps to something in the installed
version:

```bash
curl -s "https://huggingface.co/$REPO/raw/main/config.json" | head -40
ls "$LAB/venv/lib/python3.12/site-packages/sglang/srt/models/" | head -50
```

A brand-new architecture with no matching model file will fail at load no matter
what flags you pass.

## Step 1 — download

Use `hf` (the old `huggingface-cli` is deprecated on this machine). Download into
the lab so everything lives together:

```bash
cd "$LAB"
hf download "$REPO" --local-dir "models/$(basename $REPO)"
```

Large downloads are worth backgrounding. Check free disk first — these are tens
of GB.

## Step 2 — add a config to bin/serve.sh (SGLang only)

Copy the nearest existing case and adjust. The shared `COMMON` block already
carries the things that are easy to forget (PATH for the JIT compiler, fp8 KV
handling, the prefix-cache fix). A new case only needs the model path and the
memory knobs:

```bash
mymodel)  MODEL="$LAB/models/my-model"
          ARGS=( --context-length 131072 --mem-fraction-static 0.90
                 --max-mamba-cache-size 10 ) ;;
```

Starting values that are usually about right:
- `--mem-fraction-static 0.90` (raise to 0.92 for more context, watch the desktop)
- `--max-mamba-cache-size 10` — **only for hybrid GDN/mamba models.** Omit
  entirely for a dense model; there is no state pool to size.
- `--context-length` — ask for more than you expect; it truncates silently anyway

### Flags that matter, and why

| flag | why |
|---|---|
| `--kv-cache-dtype auto` | Quantized checkpoints ship fp8 KV calibration scales. Overriding discards them and can cause repetition collapse. Never set this to anything else. |
| `--mm-feature-transport cpu` | On a vision model the `cuda_ipc` default pins a flat 1024 MiB feature pool on the GPU even if you never send an image. |
| `--max-mamba-cache-size N` | Hybrid models only. ~75 MB/slot, 5 slots per concurrent request. Auto-sizing is wildly generous for single-user work. |
| `--cuda-graph-bs-decode 1 2 4` | Don't capture graphs for batch sizes the state pool makes unreachable. |
| `--default-chat-template-kwargs '{"preserve_thinking": false}'` | Reasoning models only. Stops `<think>` re-rendering from busting the prefix cache. |
| `--reasoning-parser` / `--tool-call-parser` | Must match the model family, else `<think>` blocks and tool calls arrive as raw text. |

## Step 3 — boot and read the log

```bash
THINK=medium nohup bin/serve.sh mymodel 30000 > logs/mymodel.log 2>&1 &
grep -E "Load weight end|Mamba Cache|KV Cache is allocated|Memory pool end" logs/mymodel.log
```

Confirm the numbers match your prediction. If actual context is far below what
you asked for, work through the levers in `troubleshooting.md`.

## Step 4 — verify it actually works

Booting is not working. Check generation, multi-turn, and (if applicable) vision:

```bash
qwen test
```

For a model that claims long context, also confirm it can *use* it — allocated
KV is not the same as usable attention:

```bash
"$LAB/venv/bin/python" bin/needle.py --depths 32768,102400 \
  --positions 0.1,0.5,0.9 --label mymodel
```

The 0.5 (middle) position is the classic failure mode. And measure speed rather
than assuming:

```bash
"$LAB/venv/bin/python" bin/bench.py --depths 1024,32768 --concurrency 1 \
  --gen 256 --repeats 3 --label mymodel
```

Note `bench.py` reports cold and warm separately: **cold** is true prefill,
**warm** is a prefix-cache hit. Comparing a warm number against another model's
cold number is meaningless. Use `--repeats 3` or more — with an even repeat count
the median of `[miss, hit]` is a nonsense midpoint.

## Step 5 — write down what you found

Add a row to `REPORT.md` with measured weight size, decode tok/s, actual context,
and needle results. Future-you will not remember whether a checkpoint was tried
and rejected, and re-testing costs another hour.

## On quantization formats

- **NVFP4 / W4A4** — best on Blackwell; uses the FP4 tensor cores. But check what
  fraction is *genuinely* 4-bit. Mixed checkpoints leave embeddings, `lm_head`,
  norms and MTP heads at bf16, which can be 30–50% of the file.
- **FP8** — well supported, roughly half the size of bf16.
- **AWQ / GPTQ** — supported for many architectures; verify for yours.
- **GGUF** — llama.cpp's format. SGLang's support is limited and typically
  dequantizes at load, which defeats the point (the VRAM saving is what makes it
  fast). If you have a good GGUF, running llama.cpp is usually the better answer
  than forcing it through SGLang.

## On speculative decoding — the drafter's SIZE decides everything

This was measured both ways on this box, and the result flips entirely on one
variable: **how many bytes the drafter adds per decode step.**

| drafter | size | outcome |
|---|---|---|
| SGLang in-checkpoint MTP head | 5.53 GB bf16 | **64.8 -> 50.3 tok/s**, context 132K -> 13K |
| llama.cpp GGUF MTP head (`blk.64`) | **0.21 GB quantized** | **85 -> 153 tok/s**, context unchanged |

Same technique, same model, opposite conclusion. Four SGLang variants were tried
(NEXTN 1-step and 3-step, the official cookbook EAGLE recipe on both the
flashinfer and triton backends) and all four lost — even at 98% draft acceptance,
because acceptance cannot pay for a drafter that costs a quarter of the target
model to read.

So before enabling speculation, ask **how big the draft head is and whether it is
quantized to the same precision as the target.** If it's bf16 against a 4-bit
target, expect it to lose no matter how you tune it. If it's a fraction of a GB,
expect a large win. Always measure against the no-speculation baseline — run the
benchmark twice, once with and once without.

A useful corollary: a GGUF often ships a quantized MTP head where the safetensors
build of the same model does not. That alone can be the reason to prefer
llama.cpp for a given model.
