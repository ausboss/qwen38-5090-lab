#!/usr/bin/env bash
# Optional second layer: a firewall in front of the AI services.
#
#   bin/firewall-setup.sh            show the plan, change nothing  (default)
#   bin/firewall-setup.sh --apply    actually configure and enable ufw
#
# READ THIS BEFORE --apply
# ------------------------
# This enables a default-deny-incoming firewall on the machine. The rules below
# allow SSH *before* enabling, so an existing SSH session should survive — but a
# firewall is still the classic way to lock yourself out of a remote box. If this
# machine is not in front of you, don't run --apply.
#
# Everything is reversible with:  sudo ufw disable
#
# Why you might want it
# ---------------------
# The endpoint has an API key, so the LLM itself is not open. But a scan of this
# box found these listening on 0.0.0.0 with NO authentication:
#
#     8188   ComfyUI        — can queue workflows and read/write files
#     8100   uvicorn        — unidentified
#     3389   krdpserver     — remote desktop
#     1716   kdeconnectd
#     47984/47989/47990/48010  — Sunshine game streaming
#
# ComfyUI is the one that matters: an unauthenticated ComfyUI on a shared network
# is a file-read/file-write primitive. This script does not touch those services;
# it just stops the wider network from reaching them while leaving your own LAN
# and tailnet working.
#
# What it allows
# --------------
#   - anything over tailscale0            (the tailnet is already authenticated)
#   - SSH from the LAN
#   - 30001 / 8080 / 8188 from the LAN
#   - everything outbound
# and denies other inbound. Adjust LAN_CIDR below if your subnet differs.

set -uo pipefail
LAN_CIDR="${LAN_CIDR:-$(ip -o -4 addr show scope global 2>/dev/null \
  | awk '$2!~/^(tailscale|tun|docker|br-|veth)/{print $4;exit}' \
  | awk -F/ '{split($1,o,".");print o[1]"."o[2]"."o[3]".0/"$2}')}"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

echo "LAN subnet detected: ${LAN_CIDR:-<none>}"
[ -z "$LAN_CIDR" ] && { echo "could not detect a LAN subnet; set LAN_CIDR=... and rerun" >&2; exit 1; }
echo "Tailscale iface   : $(ip -o link show tailscale0 >/dev/null 2>&1 && echo present || echo MISSING)"
echo

RULES=(
  "allow in on tailscale0"
  "allow from $LAN_CIDR to any port 22 proto tcp"
  "allow from $LAN_CIDR to any port 30001 proto tcp"
  "allow from $LAN_CIDR to any port 8080 proto tcp"
  "allow from $LAN_CIDR to any port 8188 proto tcp"
)

echo "Plan:"
echo "  sudo ufw default deny incoming"
echo "  sudo ufw default allow outgoing"
for r in "${RULES[@]}"; do echo "  sudo ufw $r"; done
echo "  sudo ufw --force enable"
echo

if [ "$APPLY" != 1 ]; then
  echo "Dry run — nothing changed. Re-run with --apply to do it."
  echo "Undo at any time with: sudo ufw disable"
  exit 0
fi

# Order matters: the allow rules go in BEFORE enable, so SSH is never briefly
# denied. ufw applies rules atomically on enable, but this ordering also means a
# half-finished run leaves the firewall off rather than on-and-blocking.
sudo ufw default deny incoming  || exit 1
sudo ufw default allow outgoing || exit 1
for r in "${RULES[@]}"; do
  # shellcheck disable=SC2086
  sudo ufw $r || { echo "rule failed: $r — NOT enabling" >&2; exit 1; }
done
sudo ufw --force enable || exit 1
echo
sudo ufw status verbose
echo
echo "Enabled. Undo with: sudo ufw disable"
