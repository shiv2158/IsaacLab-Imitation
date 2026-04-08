#!/usr/bin/env bash
# Submit parallel FastSAC G1 imitation ablations to PACE (via docker/cluster/cluster_interface.sh).
# Run from WSL after: chmod +x scripts/cluster/submit_fastsac_ablations.sh
#
# Prerequisites: docker/cluster/.env.cluster configured; singularity image on cluster.
# Each invocation syncs the workspace (costly); for many trials, batch edits or a job array
# on the cluster may be preferable later.
#
# New train.py flags (see scripts/rlopt/train.py --help):
#   --rlopt-target-tau, --rlopt-replay-size, --rlopt-feature-update-ratio,
#   --rlopt-mini-batch-size, --wandb-exp-suffix

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IFACE="${REPO_ROOT}/docker/cluster/cluster_interface.sh"

if [[ ! -x "${IFACE}" ]] && [[ -f "${IFACE}" ]]; then
  chmod +x "${IFACE}" || true
fi

COMMON=(
  --task "Isaac-Imitation-G1-v0"
  --algo FASTSAC
  --headless
)

submit_one() {
  local suffix="$1"
  shift
  echo "======== Submitting W&B exp suffix='${suffix}' extra args: $* ========"
  "${IFACE}" job base "${COMMON[@]}" "$@" --wandb-exp-suffix "${suffix}"
}

# --- Your overnight matrix (recommend submitting all in parallel if queue allows) ---

# A: Softer target update (τ=0.01). Config default uses τ=0.125 → very fast moving targets.
submit_one tau001 --rlopt-target-tau 0.01

# B: Larger replay (reduces full-buffer churn with high UTD).
submit_one replay2m --rlopt-replay-size 2000000

# C: Half the critic updates per env batch (smoother, fewer steps per hour).
submit_one fur32 --rlopt-feature-update-ratio 32

# D: Combined A+B (good single “best bet” if you only add one extra job).
submit_one tau001_r2m --rlopt-target-tau 0.01 --rlopt-replay-size 2000000

echo "[INFO] All submissions issued. Check W&B group g1_fastsac for runs named *_<suffix>."
