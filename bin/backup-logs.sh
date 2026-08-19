#!/usr/bin/env bash
# Archive the local-AI service logs.
#
#   bin/backup-logs.sh [dest_dir] [--since "2 days ago"] [--purge]
#
# Where the logs actually live
# ----------------------------
# llama-swap writes NO files of its own — it has no --log flag and holds no file
# descriptors. Everything goes to stdout, and since it runs as a systemd user
# service (StandardOutput=journal) that means journald:
#
#     journalctl --user -u llama-swap
#     journalctl --user -u open-webui
#
# It also serves an in-memory ring buffer at http://127.0.0.1:30001/logs. That
# is text/plain, capped, and lost on restart — useful for a quick look, not a
# record, so this script pulls from journald instead.
#
# The llama-server children inherit llama-swap's stdout, so their output is in
# the same journal. logs/ in this repo only holds older direct-launch runs and
# SGLang boots.
#
# What is in them
# ---------------
# Access lines (method, path, status, bytes, duration) plus model load/unload
# and health checks. Verified: no prompt or response content — grepping the
# journal for distinctive text from real conversations returns nothing. So these
# are low-sensitivity. They do reveal usage timing and which models you run.
#
# Output is chmod 600 and gzipped regardless.

set -uo pipefail
DEST="${1:-$HOME/log-backups}"
[ "${1:-}" = "--since" ] && DEST="$HOME/log-backups"
SINCE=""
PURGE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --since) SINCE="$2"; shift 2 ;;
    --purge) PURGE=1; shift ;;
    *) shift ;;
  esac
done

UNITS="llama-swap open-webui"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$DEST/$STAMP"
mkdir -p "$OUT" && chmod 700 "$DEST" "$OUT"

for u in $UNITS; do
  args=(--user -u "$u" --no-pager -o short-iso)
  [ -n "$SINCE" ] && args+=(--since "$SINCE")
  # || true: a unit with no entries yet exits non-zero, which shouldn't abort
  journalctl "${args[@]}" 2>/dev/null | gzip -9 > "$OUT/$u.log.gz" || true
  chmod 600 "$OUT/$u.log.gz"
  n=$(zcat "$OUT/$u.log.gz" 2>/dev/null | wc -l)
  printf '  %-12s %6s lines  %s\n' "$u" "$n" "$(du -h "$OUT/$u.log.gz" | cut -f1)"
done

# The in-memory buffer, if the router is up — it has lines journald may not have
# flushed yet.
if curl -s -m 5 -o /dev/null http://127.0.0.1:30001/logs 2>/dev/null; then
  curl -s -m 10 http://127.0.0.1:30001/logs | gzip -9 > "$OUT/llama-swap-ringbuffer.log.gz"
  chmod 600 "$OUT/llama-swap-ringbuffer.log.gz"
  echo "  ring buffer  captured"
fi

echo "-> $OUT"

if [ "$PURGE" = 1 ]; then
  # Only ever vacuums the journal by TIME, never rm's anything by hand — hand
  # deletion of journal files corrupts the index.
  echo "vacuuming journal older than 7 days..."
  journalctl --user --vacuum-time=7d 2>&1 | tail -2
fi

# Keep the last 30 archives.
ls -1dt "$DEST"/*/ 2>/dev/null | tail -n +31 | while read -r old; do rm -rf "$old"; done
