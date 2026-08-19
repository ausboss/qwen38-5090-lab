#!/usr/bin/env bash
# Create reusable "profiles" in Open WebUI — the equivalent of LM Studio's
# Preset dropdown. Each one is a saved system prompt + sampling params that
# appears in the model picker at the top of a chat.
#
#   bin/webui-presets.sh              # create/refresh them
#   bin/webui-presets.sh --list       # show what exists now
#   bin/webui-presets.sh --delete     # remove the ones this script made
#
# You must have an Open WebUI account first (the first account you create at
# http://127.0.0.1:8080 automatically becomes admin). This script asks for that
# email + password, uses them for a single sign-in call to get an API token, and
# never writes them anywhere. Presets are rows owned by a user, which is why
# they can't exist before an account does.
#
# By default every preset targets whichever local server is running. Override:
#   MODEL=qwen38 bin/webui-presets.sh     # force the SGLang model id

set -uo pipefail
UI=${UI:-http://127.0.0.1:8080}
LAB="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"

# --- pick the base model -----------------------------------------------------
# A preset whose base model isn't being served shows as unavailable in the
# picker, so default to whatever is actually up rather than a hardcoded guess.
detect_model() {
  if curl -s -m 3 http://127.0.0.1:30001/v1/models 2>/dev/null | grep -q qwen38-fast; then
    echo qwen38-fast
  elif curl -s -m 3 http://127.0.0.1:30000/v1/models 2>/dev/null | grep -q qwen38; then
    echo qwen38
  else
    echo ""
  fi
}
MODEL="${MODEL:-$(detect_model)}"
if [ -z "$MODEL" ]; then
  echo "No local model server is running. Start one first:" >&2
  echo "    qwen          # fast (llama.cpp)" >&2
  echo "    qwen long     # SGLang" >&2
  exit 1
fi

curl -s -m 5 "$UI/api/config" >/dev/null 2>&1 || {
  echo "Open WebUI isn't responding at $UI — start it with: qwen ui" >&2; exit 1; }

# --- auth --------------------------------------------------------------------
read -rp  "Open WebUI email: " EMAIL
read -rsp "Open WebUI password: " PASSWORD; echo

TOKEN=$(curl -s "$UI/api/v1/auths/signin" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"email":sys.argv[1],"password":sys.argv[2]}))' "$EMAIL" "$PASSWORD")" \
  | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("token",""))
except Exception: print("")')
unset PASSWORD
[ -n "$TOKEN" ] || { echo "Sign-in failed. Create the account first at $UI, then rerun." >&2; exit 1; }
echo "signed in. base model: $MODEL"

IDS="qwen38-fast qwen38-deep qwen38-vision qwen38-longdoc qwen38-code qwen38-writer"

case "${1:-}" in
  --list)
    curl -s "$UI/api/v1/models/" -H "Authorization: Bearer $TOKEN" \
      | python3 -c 'import sys,json
for m in json.load(sys.stdin):
    p=m.get("params") or {}
    sysp=(p.get("system") or "")[:48].replace("\n"," ")
    print(f"  {m[\"id\"]:<22} {m.get(\"name\",\"\"):<26} temp={p.get(\"temperature\")}  {sysp}")'
    exit 0 ;;
  --delete)
    for id in $IDS; do
      curl -s -X POST "$UI/api/v1/models/model/delete?id=$id" -H "Authorization: Bearer $TOKEN" >/dev/null
      echo "  deleted $id"
    done
    exit 0 ;;
esac

# --- create ------------------------------------------------------------------
mk() {
  local json="$1"
  local id name
  id=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["id"])' "$json")
  name=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["name"])' "$json")
  # Idempotent: delete any previous version so re-running updates cleanly.
  curl -s -X POST "$UI/api/v1/models/model/delete?id=$id" -H "Authorization: Bearer $TOKEN" >/dev/null 2>&1
  local code
  code=$(curl -s -o /tmp/owui_mk.json -w '%{http_code}' -X POST "$UI/api/v1/models/create" \
    -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d "$json")
  if [ "$code" = "200" ]; then echo "  ✓ $name"
  else echo "  ✗ $name  ($code) $(head -c 180 /tmp/owui_mk.json)"; fi
}

# Sampling notes: Qwen3.8 ships temperature 1.0 / top_k 20 / top_p 0.95 in its
# generation_config.json — that's the author's recommendation and what "Deep"
# uses. The others deliberately deviate: lower temperature where correctness or
# extraction accuracy matters more than variety.

echo "creating presets..."

mk '{"id":"qwen38-fast","base_model_id":"'"$MODEL"'","name":"Qwen3.8 · Fast",
"meta":{"description":"Quick answers, minimal preamble.","profile_image_url":"/static/favicon.png",
"capabilities":{"vision":true,"usage":true},"tags":[{"name":"local"}]},
"params":{"temperature":0.7,"top_p":0.8,"top_k":20,"max_tokens":2048,
"system":"Be direct and concise. Skip preamble and restating the question. If a short answer is complete, give a short answer."},
"is_active":true}'

mk '{"id":"qwen38-deep","base_model_id":"'"$MODEL"'","name":"Qwen3.8 · Deep",
"meta":{"description":"Hard problems. Full reasoning budget, model-default sampling.",
"capabilities":{"vision":true,"usage":true},"tags":[{"name":"local"}]},
"params":{"temperature":1.0,"top_p":0.95,"top_k":20,"max_tokens":16384,
"system":"Work through the problem carefully before answering. State your assumptions. If a question is ambiguous, say which reading you chose and why."},
"is_active":true}'

mk '{"id":"qwen38-code","base_model_id":"'"$MODEL"'","name":"Qwen3.8 · Code",
"meta":{"description":"Coding and debugging. Low temperature.",
"capabilities":{"vision":true,"usage":true},"tags":[{"name":"local"},{"name":"code"}]},
"params":{"temperature":0.2,"top_p":0.9,"top_k":20,"max_tokens":8192,
"system":"You are a careful programmer. Give working code, not sketches. Match the style of surrounding code. Point out edge cases and failure modes rather than pretending they do not exist. If you are unsure an API exists, say so instead of inventing it."},
"is_active":true}'

mk '{"id":"qwen38-vision","base_model_id":"'"$MODEL"'","name":"Qwen3.8 · Vision",
"meta":{"description":"Screenshots, diagrams, photos.",
"capabilities":{"vision":true,"usage":true,"file_upload":true},
"tags":[{"name":"local"},{"name":"vision"}]},
"params":{"temperature":0.6,"top_p":0.9,"top_k":20,"max_tokens":4096,
"system":"Describe what is actually visible. Do not guess at text that is too small or blurred to read — say it is illegible instead."},
"is_active":true}'

mk '{"id":"qwen38-longdoc","base_model_id":"'"$MODEL"'","name":"Qwen3.8 · Long doc",
"meta":{"description":"100K+ documents. Paste once, then ask many questions.",
"capabilities":{"vision":true,"usage":true,"file_upload":true},
"tags":[{"name":"local"},{"name":"128k"}]},
"params":{"temperature":0.3,"top_p":0.9,"top_k":20,"max_tokens":8192,
"system":"Answer only from the provided document. Quote the passage you relied on. If the answer is not in the document, say so plainly rather than filling the gap from general knowledge."},
"is_active":true}'

mk '{"id":"qwen38-writer","base_model_id":"'"$MODEL"'","name":"Qwen3.8 · Writing",
"meta":{"description":"Drafting and editing prose. Higher variety.",
"capabilities":{"vision":true,"usage":true},"tags":[{"name":"local"},{"name":"writing"}]},
"params":{"temperature":1.0,"top_p":0.95,"top_k":40,"max_tokens":8192,
"system":"Write in plain, direct prose. No filler openings, no summary paragraph restating what you just said. Vary sentence length. Prefer concrete detail over abstraction. Do not use em-dashes as a tic."},
"is_active":true}'

cat <<EOF

Done. Reload $UI and open the model dropdown at the top of a chat —
these appear alongside the raw models, exactly like LM Studio's preset list.

  edit one   : Workspace → Models → click it
  add your own: Workspace → Models → +   (set System Prompt + params, save)
  re-run this : safe, it replaces its own presets and leaves yours alone
  remove      : bin/webui-presets.sh --delete
  see current : bin/webui-presets.sh --list

All six point at "$MODEL". If you switch servers (qwen <-> qwen long), rerun
this so they follow, or edit each preset's base model in the UI.
EOF
