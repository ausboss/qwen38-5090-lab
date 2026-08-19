# Running Qwen3.8-27B locally — the practical guide

This is the plain-English version. [REPORT.md](REPORT.md) has the benchmark
detail if you ever want it.

## The short version

```bash
qwen          # load the default build
qwen list     # every build the endpoint offers
qwen use <id> # switch build (loads on demand, ~6s)
qwen ui       # open the chat page
qwen net      # every URL this box answers on, plus the API key
qwen status   # what's running
qwen test     # verify chat + images work
```

That's it. `qwen` works from any directory.

- **Chat page:** http://127.0.0.1:8080 (Open WebUI — already installed here)
- **API endpoint** for ComfyUI or scripts: `http://127.0.0.1:30001/v1`
  - model name: any id from `qwen list`
  - **API key: required.** `qwen key` prints it. See
    [Reaching it from other machines](#reaching-it-from-other-machines).

Both are reachable from your LAN and over Tailscale, not just this box.

### Making room for a big ComfyUI model

Two things hold GPU memory, and you need both to let go:

```bash
qwen unload     # stops the LLM AND tells ComfyUI to drop its models
                # -> frees ~26GB. The chat page stays open.
qwen            # bring the model back when you're done
```

`qwen unload` deliberately leaves Open WebUI and ComfyUI **running** — so if you
just had the model write you an image prompt, you can still read and copy it
while the GPU is free.

Measured round trip: 9.8GB free → **29.2GB free** → back to 9.8GB, chat intact.

Related:

| command | what it does |
|---|---|
| `qwen gpu` | who is holding VRAM right now, per process |
| `qwen free` | only make ComfyUI drop its models (LLM untouched) |
| `qwen stop` | stop the model only — web UI keeps running |
| `qwen down` | stop the model *and* the web UI |

**ComfyUI caches models after every generation.** That's the usual reason the LLM
suddenly refuses to start: ComfyUI is silently sitting on 18GB. `qwen gpu` shows
it, `qwen free` fixes it.

### The uncensored builds

```bash
qwen use qwen38-uncfast      # fast one, 131K context, ComfyUI still fits
qwen use qwen38-uncensored   # best fidelity, 98K context, no room for ComfyUI
```

Or just pick them in the Open WebUI dropdown — they load on demand.

| | `uncfast` | `uncensored` |
|---|---|---|
| build | mradermacher Q4_K_S | JonathanColetti Q6_K |
| weights | 15.8 GB | 22.4 GB |
| **decode** | **132.4 tok/s** | 115.6 tok/s |
| context | **131,072** | 98,304 |
| draft acceptance | 0.703 | **0.819** |
| VRAM left over | **7.4 GB** (ComfyUI fits) | 2.6 GB (it doesn't) |
| reasoning probe | 8/8 | 8/8 |
| needle (long-context recall) | **6/6** @32K+100K | 5/6 @32K+90K † |
| engaged with legit prompts | 6/6 | 6/6 |

**Use `uncfast` by default.** It's faster, has the full context, and leaves room
for image generation. Reach for `uncensored` (Q6_K) when output quality matters
more than speed — its higher draft acceptance and quantization fidelity make it
the better of the two for careful work, which is also what the build's own author
recommends for behavioural evaluation.

† The one Q6_K miss (90K, position 0.1) returned an empty answer, but re-running
that exact cell passes with the right code and `finish_reason: stop`. So it is a
flake, not a depth limit — worth knowing it can happen rather than pretending
the run was clean.

### Careful: most uncensored builds are silently broken here

Of the four uncensored Qwen3.8-27B GGUFs on this machine, **three have the MTP
draft head stripped** — they run ~40% slower and nothing warns you:

| build | MTP |
|---|---|
| mradermacher Uncensored-Aggressive Q4_K_S | present |
| JonathanColetti Uncensored Q6_K | present |
| 0bserverx Heretic-Abliterated Q4_K_M | **missing** |
| chimingw AEON-ULTIMATE Q6_K | **missing** |
| JonathanColetti `noMTP-*` (whole family) | **missing, by design** |

Abliteration re-saves the model through transformers, which drops the `mtp.*`
tensors while `config.json` still advertises them. Check any GGUF before
committing to it:

```bash
bin/gguf-inspect.py /path/to/model.gguf
```

It reports the head, the quant mix, and the predicted tok/s. The tell is
`blocks=65, nextn layers=1` (good) versus `blocks=64, nextn layers=None` (head
stripped).

Two things worth knowing about the Q6_K build specifically:

- **It's 22.4 GB, so only ~2.6 GB is left.** ComfyUI image generation will not
  fit alongside it. Run `qwen free` first, or use `qwen38-fast` for image work.
- **Context is 98K, not 131K** — the weights don't leave room for a bigger KV
  pool. Still well past 64K.

### Two setups, and when to use each

| | `qwen` (default) | `qwen long` |
|---|---|---|
| runtime | llama.cpp | SGLang |
| speed, short chat | **153 tok/s** | 65 tok/s |
| speed at 100K | **80 tok/s** | 55 tok/s |
| context | 131K | 184K |
| first reply to a **fresh** 100K doc | 49 s | **25 s** |
| port / model | 30001 / `qwen38-fast` | 30000 / `qwen38` |

**Use `qwen` for almost everything** — it's 2.4× faster to type out. Only reach
for `qwen long` if you repeatedly paste *brand new* 100K documents and the 25s
vs 49s first-token wait matters. Once a document is cached both are instant.

## How this differs from LM Studio

LM Studio is one app that does both jobs: it runs the model *and* gives you the
chat window. Here those are two separate pieces:

| | LM Studio | This setup |
|---|---|---|
| runs the model | LM Studio | `qwen` (SGLang) |
| chat window | LM Studio | Open WebUI (`qwen ui`) |
| API for ComfyUI | LM Studio's server toggle | always on at port 30000 |

The upside of the split: the API endpoint is always available whether or not
you have a chat window open, and you get ~160K of context instead of what
LM Studio typically gives you on this model.

## What's configured in Open WebUI

Set up and verified on 2026-08-19. Re-apply any time with:

```bash
pkill -f "open_webui|open-webui serve"; sleep 5
python3 bin/webui-configure.py
open-webui serve --port 8080 &
```

| feature | what you get |
|---|---|
| **Image generation** | ComfyUI + Z-Image Turbo, ~10s per 1024×1024. Toggle the image button in the chat input. |
| **Web search** | DuckDuckGo, no API key, no account |
| **Built-in tools** | search_web, fetch_url, execute_code, generate_image, memory, notes, calendar — on by default |
| **Code interpreter** | Python in your browser (Pyodide), nothing to install |
| **Long chats** | auto-compaction past 100K tokens instead of falling off a cliff |
| **Documents** | hybrid BM25 + vector search, OCR for scanned PDFs, all local/CPU |
| **10 slash-prompts** | `/rootcause` `/review` `/vram` `/comfy` `/shorter` `/eli5` `/steelman` `/checkit` `/bash` `/summarise` |

**The `qwen38-task` preset** is internal — it handles chat titles and tags with
thinking disabled. Measured 112 → 12 tokens for the same title, identical result.
Don't pick it for chat; it's wired up automatically.

> **Note:** your live database is inside the uv tool install, **not**
> `~/.open-webui/webui.db` — that path is a stale leftover from an older launch
> method and editing it does nothing. Backups are in `~/.open-webui-backups/`.

## First-time Open WebUI setup (one minute, once)

Run `qwen ui`, then in the browser:

1. **Settings → Admin Settings → Connections → OpenAI API → +**
2. URL: `http://127.0.0.1:30000/v1`
3. Key: `local` (any text works)
4. Save, reload the page. `qwen38` now appears in the model dropdown.

I deliberately did **not** auto-configure this — you have an existing
Open WebUI database and forcing settings via environment variables would have
overwritten your connections and could have disabled your login.

## System prompts

- **For one chat:** the **Controls** panel (top-right in Open WebUI) → System Prompt.
- **For everything:** Settings → General → System Prompt.
- **From code:** just a normal `system` message —

```python
messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user",   "content": "What is the capital of Japan?"},
]
```

## Logs — where they are, how to back them up, how to turn them off

**llama-swap writes no files of its own.** It has no `--log` flag and holds no
file descriptors; everything goes to stdout. Because it runs as a systemd user
service (`StandardOutput=journal`), that means journald:

```bash
journalctl --user -u llama-swap -f      # live
journalctl --user -u open-webui -n 100
```

It also serves an in-memory ring buffer at <http://127.0.0.1:30001/logs> —
`text/plain`, capped, wiped on restart. Fine for a glance, not a record.

The `llama-server` children inherit llama-swap's stdout, so they land in the
same journal. `logs/` in this repo only holds older direct-launch runs and
SGLang boots.

### What's actually in them

Access lines — method, path, status, byte count, duration — plus model
load/unload and health checks. **No prompt or response content.** Verified by
grepping the journal for distinctive text from real conversations: zero matches.
They do reveal *when* you use it and *which* models, so they're not nothing, but
they aren't transcripts.

### Backing them up

```bash
bin/backup-logs.sh                              # -> ~/log-backups/<timestamp>/
bin/backup-logs.sh /mnt/backup --since "7 days ago"
bin/backup-logs.sh --purge                      # also vacuum journal >7 days old
```

Gzipped, `chmod 600`, keeps the last 30 archives. It vacuums by *time* via
`journalctl --vacuum-time` and never deletes journal files by hand — hand
deletion corrupts the index.

### Turning them down or off

**Quieter, still useful** — in `configs/llama-swap.yaml`:

```yaml
logLevel: warn      # or error. info is the default and logs every request.
```
then `systemctl --user restart llama-swap`.

**Off entirely** — in `~/.config/systemd/user/llama-swap.service`:

```ini
StandardOutput=null
StandardError=null
```
then `systemctl --user daemon-reload && systemctl --user restart llama-swap`.
Note this also discards startup errors, so if a model then fails to load you
get no explanation. `logLevel: warn` is usually the better trade.

**Cap the journal instead** (affects all units, needs root) — in
`/etc/systemd/journald.conf`:

```ini
SystemMaxUse=200M
MaxRetentionSec=2week
```

Your journal is currently **~920 MB with no limits set**, which is worth capping
regardless of this project.


## Thinking levels

```bash
qwen think off     # no <think> block at all
qwen think 128     # hard cap: think at most ~128 tokens
qwen think low     # told to keep it brief
qwen think on      # default
qwen think long    # no guidance — rambles most (see below)
```

Restarts whichever model is loaded (~1 min) and applies to everything after.
Run `qwen think` with no argument to see this table.

### Measured on this machine

Same multi-step word problem each time, counting words inside `<think>`:

| level | thinking | total output | mechanism |
|---|---|---|---|
| `off` | **0w** | 926 tok | `--reasoning off` |
| `128` | **69w** | 356 tok | `--reasoning-budget 128` |
| `low` | 227w | 757 tok | `reasoning_effort=low` |
| `on` | 271w | 978 tok | template default (`xhigh`) |
| `long` | **435w** | 1264 tok | `reasoning_effort=medium` |

**A numeric budget is the best control** — it's the only true hard cap that still
lets the model think. At 128 it used 69 words and still got the answer right,
cutting total output by 64%.

**`long` is not a typo.** It maps to the template's `reasoning_effort=medium`,
which injects *no instruction at all*, so the model rambles more than when told
to "think carefully" (`xhigh`, the default). The template's own names are
misleading; the levels above are named for what they actually do.

**`off` isn't automatically faster.** On an easy question it's a big win. On a
hard one the model reasons in the visible answer instead — total output can grow.
It changes *where* the reasoning goes, not always how much. A numeric budget
avoids this.

### Per-request, without restarting

Works on all the llama.cpp builds (everything `qwen list` shows):

```json
{ "chat_template_kwargs": {"reasoning_effort": "low"} }
{ "chat_template_kwargs": {"enable_thinking": false} }
```

Put either in an Open WebUI preset's params to get per-profile thinking levels.
Note the top-level `reasoning_effort` field and per-request `reasoning_budget`
are both **ignored** — only the two forms above work.

> **On the SGLang path (`qwen long`) only `off` works.** SGLang inspects the chat
> template at boot, decides this model has no effort control (`effort_kwarg=None`
> in its log) and never forwards the value. The llama.cpp builds pass kwargs
> straight into the Jinja template, which is why the levels work there.

## Images

Images work — just paste or attach one in Open WebUI, or send it the standard
OpenAI way:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What's in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}
```

Verified working: `qwen test` sends a picture of a red circle and the number 42
and checks the model reads both back.

## Using it from ComfyUI or your own scripts

Anywhere that asks for an "OpenAI-compatible" endpoint:

```
Base URL : http://127.0.0.1:30001/v1
API key  : run `qwen key`
Model    : any id from `qwen list`
```

Python example:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30001/v1", api_key="lab-...")
r = client.chat.completions.create(
    model="qwen38-fast",
    messages=[{"role": "user", "content": "hello"}],
)
print(r.choices[0].message.content)
```

`GET /v1/models` lists **every** build, not just the loaded one, and asking for
one that isn't resident loads it (~6s) instead of erroring. So a script can name
`qwen38-uncfast` and it just works, whatever was running before.

## Reaching it from other machines

`qwen net` prints everything below for your current addresses.

The router listens on `0.0.0.0:30001` and Open WebUI on `0.0.0.0:8080`, so both
answer on three addresses — localhost, your LAN IP, and your Tailscale IP:

```
http://127.0.0.1:30001/v1          this box
http://192.168.1.29:30001/v1       anything on the wifi
http://100.120.215.57:30001/v1     anything on the tailnet, from anywhere
http://5090:30001/v1               same, via MagicDNS — survives IP changes
```

Prefer the MagicDNS name (`5090`) in configs you keep: Tailscale IPs are stable
in practice but the name is stable by contract.

### The API key

Because it listens on all interfaces, the endpoint requires a bearer token.
Without one it returns **401**, so an unattended device on your network can't
quietly spend your GPU.

```bash
qwen key                                  # print it
curl -H "Authorization: Bearer $(qwen key)" http://5090:30001/v1/models
bin/net-setup.sh --rotate                 # mint a new one, re-wire Open WebUI
```

It lives in `configs/secrets/api-key.txt`, which is gitignored. The model
definitions in `configs/llama-swap.yaml` stay tracked — llama-swap merges the
two with `--config-dir`.

> **Trap 1:** in llama-swap's config the key is `apiKeys:` — a *list*. The
> obvious singular `apiKey:` is accepted, logs nothing, and silently disables
> auth entirely. Verified on v250: singular returned 200 for a deliberately
> wrong key; plural returned 401.
>
> **Trap 2:** if `configs/secrets/` exists but is *empty*, llama-swap reads that
> as "no apiKeys configured" and starts wide open on `0.0.0.0` — no warning in
> the log. The systemd unit has an `ExecStartPre` that refuses to start in that
> case, so the failure is loud instead of silent. If you run llama-swap by hand,
> you don't get that guard.

Either way, the check is the same one-liner — you want **401**:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:30001/v1/models
```

### What is *not* exposed

- **The `llama-server` children stay on `127.0.0.1`.** Only the router is
  reachable from off-box, so there's one door and it needs a key. Don't change
  `--host` in the `common` macro — that would put an unauthenticated
  `llama-server` on a random high port every time a model loads.
- **SGLang (`qwen long`) is loopback-only** and has no auth. Tunnel it if you
  need it remotely:
  ```bash
  ssh -N -L 30000:127.0.0.1:30000 ausboss@5090
  ```
- **Open WebUI has its own login** — the API key doesn't apply to it.

### Optional: a firewall

ComfyUI (8188) listens on all interfaces with **no authentication**, and an
unauthenticated ComfyUI is a file read/write primitive. Same for a few other
things on this box (RDP on 3389, Sunshine on 47984+). That's fine on a trusted
home network and not fine on a shared one.

```bash
bin/firewall-setup.sh            # show the plan, change nothing
bin/firewall-setup.sh --apply    # allow tailnet + LAN, deny the rest
```

Read the header of that script before `--apply` — it enables a default-deny
firewall. It allows SSH before enabling, but don't run it on a box you can't
physically reach. Undo with `sudo ufw disable`.

## Things that will otherwise confuse you

- **The first message after starting is slow** (~40s). That's a one-time GPU
  warmup. `qwen` now burns it during startup, so you shouldn't see it — but if
  you start the server some other way, that's what it is.
- **Pasting a huge document takes ~25 seconds** before the reply starts. Reading
  100K tokens genuinely takes that long. *Follow-up questions about the same
  document are instant* (~0.5s) because it's cached. So paste once, then ask
  many questions — don't re-paste.
- **ComfyUI and this can't both run.** ComfyUI wants ~12 GB of VRAM and this
  uses nearly all 32 GB. Run `qwen stop` before starting ComfyUI, and vice
  versa. (If you'd rather keep some room free, `qwen safe` uses ~2 GB less at
  the cost of context.)
- **Short answers may look truncated** if you set a low `max_tokens` — the
  thinking counts against it. Give it at least ~800.

## Commands

`qwen help` prints this too.

**Models**

| command | what it does |
|---|---|
| `qwen` | load the default build (`qwen38-fast`) |
| `qwen use <id>` | load a specific build — see `qwen list` |
| `qwen list` | every build the endpoint offers, `*` marks the loaded one |
| `qwen long` / `qwen safe` | SGLang instead, port 30000 (bigger context) |

**GPU**

| command | what it does |
|---|---|
| `qwen unload` | free all model VRAM — LLM *and* ComfyUI. Web UI stays up |
| `qwen stop` | unload the LLM only |
| `qwen free` | tell ComfyUI to drop its models only |
| `qwen down` | unload the LLM *and* stop the web UI |
| `qwen gpu` | who's holding VRAM right now, per process |

**Network**

| command | what it does |
|---|---|
| `qwen net` | every URL this box answers on, plus the API key |
| `qwen key` | print the API key alone, for scripts |

**Checking**

| command | what it does |
|---|---|
| `qwen status` | what's running, plus GPU |
| `qwen test` | chat + multi-turn + image, end to end |
| `qwen ui` | open the chat page |
| `qwen think` | explains how to change reasoning depth |

`qwen think` no longer takes a level — reasoning depth is a per-model or
per-request choice now, not a restart. It prints the options.
