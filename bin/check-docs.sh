#!/usr/bin/env bash
# Catch documentation that has drifted away from the code.
#
#   bin/check-docs.sh
#
# Why this exists
# ---------------
# The docs once carried nine references to `qwen` subcommands that had been
# removed in a rewrite (uncfast, uncensored, unc, uf, reload, and `qwen think
# <level>`), plus "API key: anything at all, it isn't checked" after auth was
# turned on. Nothing failed — the commands just silently did nothing, and a
# reader following them was misled. Prose has no compiler, so this is the
# compiler.
#
# It checks three things:
#   1. every `qwen <word>` in the docs is a real subcommand
#   2. no doc still claims the endpoint is unauthenticated
#   3. the skill symlinks still point into this repo (one source of truth)
#
# Exit code is nonzero if anything is wrong, so it works in a pre-commit hook.

set -uo pipefail
LAB="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$LAB" || exit 1
FAIL=0

DOCS=(README.md USAGE.md REPORT.md NOTES.md skills/qwen38-local/SKILL.md
      skills/qwen38-local/references/*.md)

# --- 1. subcommands ----------------------------------------------------------
# Pull the case labels straight out of the script so this can never go stale
# the way a hardcoded list would.
IMPL="$(sed -n '/^case "${1:-load}"/,/^esac/p' bin/qwen \
        | grep -oE '^  [a-z|"-]+\)' | tr -d ' )"' | tr '|' '\n' | grep -v '^$' | sort -u)"

# Words that follow `qwen` in prose but aren't subcommands.
IGNORE='^(is|are|was|were|the|a|an|and|or|to|for|with|it|its|this|that|then|works|serves|uses|list|when|if|so|on|in|at|by|as|of|from|help|command|commands|does|do|will|can|run|runs|now|also|only|but|not|no|you|your|we|they|he|she|via|per|vs|versus|model|models|build|builds|first|again|instead|here|there|both|each|all|any|some|more|less|than|like|just|even|still|yet|already|never|always)$'

echo "== qwen subcommands referenced in docs =="
BAD=0
while read -r cmd; do
  [ -z "$cmd" ] && continue
  grep -qxF "$cmd" <<<"$IMPL" && continue
  echo "$cmd" | grep -qE "$IGNORE" && continue
  echo "  MISSING: 'qwen $cmd' is documented but not implemented"
  grep -rn "qwen $cmd" "${DOCS[@]}" 2>/dev/null | sed 's/^/      /' | head -4
  BAD=1
done < <(grep -rhoE '\bqwen [a-z][a-z0-9-]*' "${DOCS[@]}" 2>/dev/null \
         | awk '{print $2}' | sort -u)
[ "$BAD" = 0 ] && echo "  ok — all referenced subcommands exist" || FAIL=1

# --- 2. stale auth claims ----------------------------------------------------
echo "== stale auth claims =="
# The endpoint used to accept any key. If that text survives anywhere it is now
# actively wrong and will send someone debugging a 401 in the wrong direction.
#
# Deliberately does NOT match a bare "no auth" / "no authentication": SGLang on
# 30000 and ComfyUI on 8188 genuinely have none, and saying so is correct. Only
# unambiguous claims about the *router* belong here, or the check cries wolf and
# gets ignored — which is worse than not having it.
PAT='any API key|anything at all|api_key="local"|apiKey": "local"|key is ignored|isn.t checked'
if grep -rniE "$PAT" "${DOCS[@]}" 2>/dev/null; then
  echo "  ^ these claim the endpoint is open; it requires a bearer token"
  FAIL=1
else
  echo "  ok — none found"
fi

# --- 3. skill symlinks -------------------------------------------------------
echo "== skill symlinks =="
CANON="$LAB/skills/qwen38-local"
for p in "$HOME/.claude/skills/qwen38-local" "$HOME/.pi/agent/skills/qwen38-local"; do
  short="${p/$HOME/\~}"
  if [ ! -e "$p" ]; then
    echo "  MISSING: $short"; FAIL=1
  elif [ ! -L "$p" ]; then
    echo "  NOT A SYMLINK: $short is a real directory — it will drift"; FAIL=1
  elif [ "$(readlink -f "$p")" != "$CANON" ]; then
    echo "  WRONG TARGET: $short -> $(readlink -f "$p")"; FAIL=1
  else
    echo "  ok  $short"
  fi
done

echo
[ "$FAIL" = 0 ] && echo "docs look consistent" || echo "problems found (see above)"
exit "$FAIL"
