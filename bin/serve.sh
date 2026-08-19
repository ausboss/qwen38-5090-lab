#!/usr/bin/env bash
# Qwen3.8-27B on SGLang / RTX 5090 (32GB) — config launcher.
#
# Usage:  bin/serve.sh <config> [port]
# Configs: ctx128 | ctx128mtp | ctx200 | ctx256
#
# Design notes (verified against `sglang serve --help`, sglang 0.5.17):
#   * Model is hybrid: 64 layers = 16 full-attention + 48 Gated DeltaNet (GDN).
#     KV cache exists only on the 16 attention layers -> 32 KB/token at fp8.
#     The 48 GDN layers hold a fixed-size recurrent state (~79 MB/req at bf16)
#     that does NOT grow with context length. Long context is therefore cheap.
#   * NEVER pass --kv-cache-dtype anything but auto. The NVFP4 checkpoint ships
#     fp8 KV calibration scales; overriding discards them.
#   * ReplaySSM spec-verify requires a linear chain: --speculative-eagle-topk 1.
#     (--enable-gdn-replayssm-spec is a deprecated alias for the linear one.)
#   * --enable-linear-replayssm (no -spec) is a *batch>=64* decode optimization.
#     Deliberately NOT used here: this box is a single-user agent, batch 1-4.
#   * Vision tower is left unloaded (--enable-multimodal is opt-in) to save VRAM.

set -uo pipefail

LAB="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
MODEL="$LAB/models/Qwen3.8-27B-NVFP4"
PY="$LAB/venv/bin/python"

# FlashInfer JIT-compiles attention kernels on first boot and shells out to
# `ninja` + `nvcc`. We invoke venv/bin/python directly (no venv activation), so
# neither is on PATH by default -- without this the server dies with
# "FileNotFoundError: ninja".
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$LAB/venv/bin:$CUDA_HOME/bin:$PATH"
# Persist the JIT cache so the multi-minute kernel build happens only once.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$LAB/.flashinfer}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

CONFIG="${1:?usage: serve.sh <vision|safe|ctx64mtp|ctx32mtp> [port]}"
PORT="${2:-30000}"

# Thinking depth, applied to every request unless the client overrides it.
#   xhigh (model default) | medium | low | off
# preserve_thinking:false is ALWAYS set -- without it the model re-renders empty
# <think> blocks for every prior turn, which changes the prompt prefix, misses
# the cache, and turns a 0.5s reply into a 25s one at long context.
THINK="${THINK:-medium}"
case "$THINK" in
  off)   TEMPLATE_KWARGS='{"preserve_thinking": false, "enable_thinking": false}' ;;
  xhigh|medium|low)
         TEMPLATE_KWARGS="{\"preserve_thinking\": false, \"reasoning_effort\": \"$THINK\"}" ;;
  *)     echo "THINK must be one of: xhigh medium low off" >&2; exit 2 ;;
esac

# Flags shared by every config.
COMMON=(
  --model-path "$MODEL"
  --served-model-name qwen38
  --host 127.0.0.1 --port "$PORT"
  --attention-backend flashinfer
  --kv-cache-dtype auto          # honors the checkpoint's fp8 KV scales
  --mamba-ssm-dtype bfloat16     # halves GDN state vs the config's float32
  --chunked-prefill-size 4096
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs "$TEMPLATE_KWARGS"

  # --- VRAM reclaimed for context (measured on the first ctx128 boot) ---
  # This is a VLM. Serving it as a text agent, the vision path is pure waste:
  #   --language-only         skips loading the vision tower entirely
  #   --mm-feature-transport  the cuda_ipc default reserves a flat 1024 MiB
  #                           feature pool on GPU 0; cpu transport frees it
  # cpu transport keeps images working but avoids the 1024 MiB GPU feature pool
  # the cuda_ipc default reserves. Slightly slower per image, 1 GB more context.
  --mm-feature-transport cpu

  # The GDN state pool caps max_running_requests at 3 (see --max-mamba-cache-size
  # below), but SGLang still captures decode CUDA graphs for bs up to 24. Those
  # graphs can never be used -- they only cost boot time and VRAM. Capture the
  # batch sizes we can actually reach.
  --cuda-graph-bs-decode 1 2 4
)

# Speculative decoding via the in-checkpoint MTP head (mtp_num_hidden_layers=1).
SPEC=(
  --speculative-algorithm NEXTN
  --speculative-num-steps 3
  --speculative-eagle-topk 1     # linear chain — required by ReplaySSM
  --speculative-num-draft-tokens 4
  --enable-linear-replayssm-spec # verify buffers -> fixed ring, not per-draft
)

# The GDN state pool costs ~75 MB per slot and 5 slots per in-flight request.
# Left to auto-size it grabbed 32 slots / 2.41 GB -- enough for 6 concurrent
# requests, which is meaningless for a single-user agent and is 2.4 GB not spent
# on context. Pinning it low is the single biggest context lever after the
# vision tower. 15 slots = 3 concurrent requests.
# MEASURED weight cost, which drives every choice below:
#   target model        20.21 GB
#   + MTP draft head     5.53 GB   <-- 1 layer, but its own bf16 embed + lm_head
#                                       (vocab 248,320 x 5,120). ~170K tokens of KV.
# On a 32 GB card, MTP and 128K context cannot coexist. Pick one.
#
# Speculative decoding also resets max_running_requests to 48, which re-explodes
# the GDN pool -- it MUST be pinned explicitly alongside --max-mamba-cache-size.
# NOTE: --language-only was tested and removed. It saved nothing measurable
# (20.12 GB with the vision tower vs 20.15 GB without) while disabling images.
# Vision is effectively free on this checkpoint, so every config keeps it.
case "$CONFIG" in
  # --- daily drivers: images enabled, no MTP ---
  # vision = maximum context (~160K), runs the GPU nearly full
  # safe   = ~130K context, leaves ~2 GB so other GPU apps still launch
  vision)     ARGS=( --context-length 204800 --mem-fraction-static 0.92
                     --max-mamba-cache-size 10 ) ;;
  safe)       ARGS=( --context-length 131072 --mem-fraction-static 0.90
                     --max-mamba-cache-size 15 ) ;;

  # legacy aliases kept so REPORT.md's commands still work
  ctx128)     ARGS=( --language-only --context-length 131072
                     --mem-fraction-static 0.90 --max-mamba-cache-size 15 ) ;;
  ctx200)     ARGS=( --language-only --context-length 204800
                     --mem-fraction-static 0.92 --max-mamba-cache-size 10 ) ;;

  # --- speed-first (MTP): context sacrificed to fit the 5.53 GB draft head ---
  ctx64mtp)   ARGS=( --context-length 65536 --mem-fraction-static 0.95
                     --max-mamba-cache-size 5 --max-running-requests 1
                     "${SPEC[@]}" ) ;;
  ctx32mtp)   ARGS=( --context-length 32768 --mem-fraction-static 0.95
                     --max-mamba-cache-size 5 --max-running-requests 1
                     "${SPEC[@]}" ) ;;
  # Cheaper draft: 1 step instead of 3. Each draft step re-reads the 5.53 GB
  # bf16 head, so fewer steps = less draft cost, at the price of a shorter
  # accepted run. Tests whether 3-step drafting was the reason MTP lost.
  mtp1)       ARGS=( --context-length 32768 --mem-fraction-static 0.95
                     --max-mamba-cache-size 5 --max-running-requests 1
                     --speculative-algorithm NEXTN
                     --speculative-num-steps 1 --speculative-eagle-topk 1
                     --speculative-num-draft-tokens 2
                     --enable-linear-replayssm-spec ) ;;

  # --- the OFFICIAL SGLang cookbook RTX 5090 recipes -------------------------
  # Source: docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B.md
  #   curl -sL <that url> returns the full MDX with every recipe cell inline.
  # Differences from our earlier (losing) MTP attempt, all of which matter:
  #   * EAGLE, not NEXTN
  #   * --mamba-ssm-dtype float32 -- ReplaySSM auto-selects fp32; forcing bf16
  #     logs a state-drift warning
  #   * --mem-fraction-static 0.94 at fp32 (0.92 at bf16), not 0.95
  #   * --mamba-radix-cache-strategy extra_buffer, --cuda-graph-max-bs 1
  # Repeated flags override the COMMON block (argparse takes the last one).
  eagle)      ARGS=( --mem-fraction-static 0.94 --context-length 131072 --max-mamba-cache-size 5
                     --mamba-ssm-dtype float32
                     --mamba-radix-cache-strategy extra_buffer
                     --max-running-requests 1 --cuda-graph-max-bs 1
                     --cuda-graph-bs-decode 1
                     --chunked-prefill-size 2048
                     --speculative-algorithm EAGLE
                     --speculative-num-steps 3
                     --speculative-eagle-topk 1
                     --speculative-num-draft-tokens 4
                     --enable-linear-replayssm-spec ) ;;

  # The cookbook states MTP on the FlashInfer backend needs a build whose
  # prefill plan accepts `uniform_q_len` (newer than 0.6.15.post1 -- which is
  # exactly what is installed here; the symbol is absent from the package).
  # Its prescribed workaround is to run the spec path on triton.
  eagle-triton) ARGS=( --mem-fraction-static 0.94 --context-length 131072 --max-mamba-cache-size 5
                     --attention-backend triton
                     --mamba-ssm-dtype float32
                     --mamba-radix-cache-strategy extra_buffer
                     --max-running-requests 1 --cuda-graph-max-bs 1
                     --cuda-graph-bs-decode 1
                     --chunked-prefill-size 2048
                     --speculative-algorithm EAGLE
                     --speculative-num-steps 3
                     --speculative-eagle-topk 1
                     --speculative-num-draft-tokens 4
                     --enable-linear-replayssm-spec ) ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2 ;;
esac

echo "=== serving config '$CONFIG' on port $PORT ==="
printf '%s\n' "${COMMON[@]}" "${ARGS[@]}" | paste -sd' ' -
echo "======================================"

exec "$PY" -m sglang.launch_server "${COMMON[@]}" "${ARGS[@]}"
