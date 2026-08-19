#!/usr/bin/env bash
# Put the endpoint on the network, with a key.
#
#   bin/net-setup.sh              set it up (idempotent — keeps an existing key)
#   bin/net-setup.sh --rotate     mint a NEW key and re-wire everything to it
#   bin/net-setup.sh --show       print the current key and URLs
#
# What "on the network" means here
# --------------------------------
# llama-swap listens on 0.0.0.0:30001, so the OpenAI-compatible endpoint answers
# on localhost, the LAN address, and the Tailscale address — same base URL shape
# for all three. Open WebUI does the same on :8080.
#
# The llama-server children still bind 127.0.0.1 (see the `common` macro in
# configs/llama-swap.yaml). That is deliberate: only the router is reachable from
# off-box, so there is one door and it needs a key. Do not "helpfully" change the
# children to 0.0.0.0 — it would expose an unauthenticated llama-server on a
# random high port every time a model loads.
#
# Why a key at all
# ----------------
# An LLM endpoint with no auth on 0.0.0.0 means any device on the wifi can spend
# your GPU, and Tailscale means any device on the tailnet can too. llama-swap
# enforces `apiKeys` on everything except /health.
#
# TRAP: the singular `apiKey:` is silently ignored — the config loads, the server
# starts, and every request is accepted. It must be the plural list form:
#     apiKeys:
#       - "..."
# Verified against llama-swap v250: singular returned 200 for a wrong key,
# plural returned 401.
#
# The key lives in configs/secrets/ (gitignored) rather than in
# configs/llama-swap.yaml so the model definitions stay tracked. llama-swap
# merges both with --config-dir.

set -uo pipefail
LAB="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
SEC="$LAB/configs/secrets"
KEYFILE="$SEC/api-key.txt"
AUTHFILE="$SEC/auth.yaml"
DB="$(find "$HOME/.local/share/uv/tools/open-webui" -name webui.db 2>/dev/null | head -1)"

MODE="${1:-setup}"

gen_key() { echo "lab-$(head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 32)"; }

show() {
  echo
  "$LAB/bin/qwen" net
}

case "$MODE" in
  --show) show; exit 0 ;;
esac

mkdir -p "$SEC" && chmod 700 "$SEC"

if [ "$MODE" = "--rotate" ] || [ ! -s "$KEYFILE" ]; then
  KEY="$(gen_key)"
  printf '%s\n' "$KEY" > "$KEYFILE"
  printf 'apiKeys:\n  - "%s"\n' "$KEY" > "$AUTHFILE"
  chmod 600 "$KEYFILE" "$AUTHFILE"
  echo "minted a new key"
else
  KEY="$(cat "$KEYFILE")"
  # Repair the yaml if it drifted from the key file.
  printf 'apiKeys:\n  - "%s"\n' "$KEY" > "$AUTHFILE"
  chmod 600 "$KEYFILE" "$AUTHFILE"
  echo "using existing key"
fi

# --- Open WebUI --------------------------------------------------------------
# Its config is PersistentConfig-cached in memory, so a running server will
# overwrite these rows on shutdown. Stop first, write, start.
if [ -n "$DB" ] && [ -f "$DB" ]; then
  WAS_UP=0
  systemctl --user is-active --quiet open-webui && WAS_UP=1
  [ "$WAS_UP" = 1 ] && { echo "stopping open-webui to write config..."; systemctl --user stop open-webui; sleep 3; }

  cp -a "$DB" "$DB.bak-$(date +%Y%m%d-%H%M%S)"
  python3 - "$DB" "$KEY" <<'PY'
import json, sqlite3, sys, time
db = sqlite3.connect(sys.argv[1]); key = sys.argv[2]
def get(k, d=None):
    r = db.execute("select value from config where key=?", (k,)).fetchone()
    return json.loads(r[0]) if r else d
def put(k, v):
    db.execute("update config set value=?,updated_at=? where key=?",
               (json.dumps(v), int(time.time()), k))
urls = get("openai.api_base_urls", []) or []
keys = get("openai.api_keys", []) or []
while len(keys) < len(urls): keys.append("")
hits = 0
for i, u in enumerate(urls):
    if ":30001" in u:
        keys[i] = key; hits += 1
put("openai.api_keys", keys); db.commit()
print(f"  open-webui: updated {hits} connection(s) to the router")
PY
  [ "$WAS_UP" = 1 ] && { systemctl --user start open-webui; echo "  restarted open-webui"; }
else
  echo "  (open-webui db not found — set the key by hand in Settings > Connections)"
fi

# --- router ------------------------------------------------------------------
if systemctl --user list-unit-files llama-swap.service >/dev/null 2>&1; then
  systemctl --user restart llama-swap
  sleep 3
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
         http://127.0.0.1:30001/v1/models)
  nokey=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30001/v1/models)
  echo "  router: with key -> $code, without key -> $nokey  (want 200 / 401)"
  [ "$nokey" = "200" ] && echo "  !! endpoint is NOT enforcing auth — check $AUTHFILE uses apiKeys (plural)"
fi

show
