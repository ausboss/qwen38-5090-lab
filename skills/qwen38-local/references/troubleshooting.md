# Failure modes, and what they actually mean

Each entry here cost real debugging time on this machine. The error text rarely
points at the true cause, so match on the symptom.

## "FileNotFoundError: ninja" during boot

FlashInfer JIT-compiles attention kernels on first use and shells out to `ninja`
and `nvcc`. Invoking `venv/bin/python` directly does **not** activate the venv, so
neither is on `PATH`.

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$LAB/venv/bin:$CUDA_HOME/bin:$PATH"
```

`bin/serve.sh` already does this. If you write a new launcher, carry it over.

## Boot appears to hang for several minutes with no log output

Usually not a hang. On a fresh install or after an SGLang/FlashInfer upgrade it
is compiling CUTLASS FP4 GEMM kernels for `sm_120f`. Confirm:

```bash
pgrep -af 'ninja|nvcc|cc1plus' | head
```

Four `cc1plus` at 100% CPU means it's working. Let it finish. `FLASHINFER_WORKSPACE_BASE`
is pinned to `$LAB/.flashinfer` so this cost is paid once, not per boot.

## "Not enough GPU memory for hybrid (mamba/linear-attention) state cache"

The GDN state pool couldn't get a single slot. Almost always because speculative
decoding is on: it silently resets `max_running_requests` to 48, which explodes
the pool. Pin both:

```
--max-running-requests 1 --max-mamba-cache-size 5
```

## "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=X"

Weights plus (if enabled) the draft head exceeded the static budget. The message
helpfully names the minimum viable value. But before raising it, ask whether the
config makes sense at all — on a 32 GB card this usually means speculative
decoding is on and its draft head is too big to coexist with real context.

## Context is far smaller than requested, with no error

Expected behaviour — SGLang truncates silently. See `memory-model.md` Rule 4.
Check `grep "KV Cache is allocated" logs/*.log`. To get more:

1. `--max-mamba-cache-size` down (biggest lever; ~75 MB per slot)
2. `--mm-feature-transport cpu` (frees a flat 1024 MiB pool on VLMs)
3. `--mem-fraction-static` up by 0.02 at a time — watch the desktop
4. Fewer CUDA graph batch sizes: `--cuda-graph-bs-decode 1 2 4`

## `pkill -f 'sglang.launch_server'` kills your own shell

The pattern matches the `bash -c` command line running the pkill. Use the bracket
trick, which matches the process but not the pattern text itself:

```bash
pkill -f 'sglang.launch_serve[r]'
```

## First request after boot takes ~40 seconds, then everything is fast

One-off kernel autotune on the decode path. Burn it with a throwaway request
during startup so the user never sees it:

```bash
curl -s -o /dev/null http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

## Long prompts are slow every single time

Prefill at 100K genuinely takes ~25 s; a *repeat* of the same prefix should take
~0.5 s. If it never gets fast, the prefix cache is being missed. The usual cause
on a reasoning model is the chat template re-rendering `<think>` blocks for prior
turns, which changes the prefix. Fix:

```
--default-chat-template-kwargs '{"preserve_thinking": false}'
```

## Answers look truncated mid-sentence

Thinking tokens count against `max_tokens`. A reasoning model can spend 800+
tokens thinking before it emits anything visible. Give it ≥1000, or disable
thinking.

## A symlinked launcher writes logs to the wrong directory

`dirname "${BASH_SOURCE[0]}"` resolves the symlink's location, not the script's.
Resolve first:

```bash
LAB="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
```

## Model won't stop / VRAM still held after stopping

`nvidia-smi` can show stale usage for a few seconds during teardown. Confirm with
`pgrep -f 'sglang.launch_serve[r]'` rather than trusting the memory figure. If a
process really is stuck, `pkill -9 -f 'sglang.launch_serve[r]'`.

## Checking whether anything is actually wrong

```bash
qwen status                         # up? how much context? GPU?
qwen test                           # chat + multi-turn + image, end to end
grep -E "KV Cache is allocated|Load weight end|Memory pool end" logs/<cfg>.log
grep -iE "error|Traceback|OutOfMemory|ValueError" logs/<cfg>.log | grep -v server_args
```

## "401 Unauthorized" from the endpoint

Expected. llama-swap listens on `0.0.0.0:30001` so it works over LAN and
Tailscale, so it requires a bearer token.

```bash
curl -H "Authorization: Bearer $(qwen key)" http://127.0.0.1:30001/v1/models
```

Python tools in `bin/` get it automatically from `bin/labauth.py`. If something
you wrote 401s, it is hardcoding a key — read it from
`configs/secrets/api-key.txt` instead.

**The inverse is the dangerous one.** If the endpoint answers *without* a key,
auth is off and anything on the network can use the GPU:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:30001/v1/models   # want 401
```

The usual cause is `apiKey:` (singular) in the config instead of `apiKeys:` (a
list). Singular is accepted, logs nothing, and disables auth entirely — verified
on v250. Fix with `bin/net-setup.sh`.

## Open WebUI restart-loops / "address already in use"

Its uvicorn worker can outlive a `systemctl --user restart` and keep holding
:8080, so the new instance can't bind and the unit retries forever, burning real
CPU. Check for an orphan and kill it **by PID**:

```bash
ss -tlnp | grep :8080
systemctl --user stop open-webui
kill <pid>
systemctl --user start open-webui
```

Never `pkill -f` a pattern here — it matches your own command line and kills the
shell running it.
