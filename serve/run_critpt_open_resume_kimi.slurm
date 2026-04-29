#!/bin/bash
# Resume OpenDirac CritPt batch against an existing vLLM serve job (e.g. local Kimi-K2.6).
#
# Submit from the repository root so log paths resolve:
#   cd /path/to/open-dirac && sbatch serve/run_critpt_open_resume_kimi.slurm
#
# Required environment (at sbatch / job time):
#   SERVE_JOB   Slurm job id of the vLLM serve allocation
#   RESUME_DIR  Path to a prior run output dir for --resume (submission JSONs tree)
#
#SBATCH --job-name critpt-kimi-open-resume
#SBATCH --partition hopper-prod
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=64G
#SBATCH --time 24:00:00
#SBATCH --output serve/logs/slurm/%x-%j.out
#SBATCH --error serve/logs/slurm/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
  REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
fi

SERVE_JOB="${SERVE_JOB:?set SERVE_JOB to vLLM serve Slurm id}"
RESUME_DIR="${RESUME_DIR:?set RESUME_DIR to CritPt --resume output directory}"

[[ -f "${HOME}/.bashrc" ]] && source "${HOME}/.bashrc"
cd "${REPO_ROOT}"

END="${REPO_ROOT}/serve/logs/vllm/${SERVE_JOB}/endpoint.env"
echo "Waiting for ${END}..."
for _ in $(seq 1 480); do
  [[ -f "${END}" ]] && break
  sleep 30
done
[[ -f "${END}" ]] || { echo "missing ${END}"; exit 1; }

# shellcheck disable=SC1090
source "${END}"
export VLLM_BASE_URL="${BASE_URL}"
HEALTH="http://${HEAD_IP}:${PORT:-8000}/health"
echo "Waiting for vLLM at ${HEALTH}..."
WAIT=0
until curl -sf "${HEALTH}" >/dev/null; do
  [[ ${WAIT} -ge 14400 ]] && { echo "health timeout"; exit 1; }
  sleep 15
  WAIT=$((WAIT + 15))
done
echo "vLLM ready after ${WAIT}s; resuming OpenDirac CritPt batch..."

exec uv run --no-sync python scripts/run_critpt_open_dirac.py \
  --resume "${RESUME_DIR}" --timeout 21600 --concurrency 2
