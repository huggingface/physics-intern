#!/bin/bash
# Resume an OpenDirac CritPt batch against an existing local vLLM Slurm serve job.
#
# Submit from the repository root so log paths resolve:
#   cd /path/to/open-dirac && sbatch serve/run_critpt_open_resume.slurm
#
# Required at sbatch/job time (e.g. --export or #SBATCH --export):
#   SERVE_JOB   Slurm job id of the vLLM serve allocation (see serve/logs/vllm/<id>/)
#
#   RESUME_DIR  Path to a prior CritPt output directory for scripts/run_critpt_open_dirac.py --resume
#
# Optional:
#   CRITPT_TIMEOUT             Per-problem timeout seconds (default: 21600)
#   CRITPT_CONCURRENCY         Max parallel problems (default: 10, same as run_critpt_open_dirac.py)
#   CRITPT_HEALTH_WAIT_SEC     Max seconds to wait for /health after endpoint.env exists (default: 14400)
#   CRITPT_HEALTH_POLL_SLEEP   Seconds between /health polls (default: 15)
#   CRITPT_ENDPOINT_POLL_MAX   Max waits for endpoint.env to appear (default: 480; 30s each)
#   CRITPT_ENDPOINT_POLL_SLEEP Seconds between endpoint.env polls (default: 30)
#   CRITPT_RESUME_DRY_RUN      If 1, exit 0 after /health succeeds (for CI — does not run CritPt)
#   CRITPT_SKIP_BASHRC         If 1, do not source ~/.bashrc (cleaner automated tests)
#
# Override Slurm resources or partition when submitting, e.g.:
#   sbatch --partition=gpu --cpus-per-task=8 serve/run_critpt_open_resume.slurm
#
#SBATCH --job-name critpt-open-resume
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

SERVE_JOB="${SERVE_JOB:?set SERVE_JOB to the vLLM serve Slurm job id}"
RESUME_DIR="${RESUME_DIR:?set RESUME_DIR to the CritPt --resume output directory}"

CRITPT_TIMEOUT="${CRITPT_TIMEOUT:-21600}"
CRITPT_CONCURRENCY="${CRITPT_CONCURRENCY:-10}"
CRITPT_HEALTH_WAIT_SEC="${CRITPT_HEALTH_WAIT_SEC:-14400}"
CRITPT_HEALTH_POLL_SLEEP="${CRITPT_HEALTH_POLL_SLEEP:-15}"
CRITPT_ENDPOINT_POLL_MAX="${CRITPT_ENDPOINT_POLL_MAX:-480}"
CRITPT_ENDPOINT_POLL_SLEEP="${CRITPT_ENDPOINT_POLL_SLEEP:-30}"

if [[ "${CRITPT_SKIP_BASHRC:-}" != "1" ]] && [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc"
fi
cd "${REPO_ROOT}"

END="${REPO_ROOT}/serve/logs/vllm/${SERVE_JOB}/endpoint.env"
echo "Waiting for ${END}..."
for _ in $(seq 1 "${CRITPT_ENDPOINT_POLL_MAX}"); do
  [[ -f "${END}" ]] && break
  sleep "${CRITPT_ENDPOINT_POLL_SLEEP}"
done
[[ -f "${END}" ]] || { echo "missing ${END}"; exit 1; }

# shellcheck disable=SC1090
source "${END}"
export VLLM_BASE_URL="${BASE_URL}"
HEALTH="http://${HEAD_IP}:${PORT:-8000}/health"
echo "Waiting for vLLM at ${HEALTH}..."
WAIT=0
until curl -sf "${HEALTH}" >/dev/null; do
  [[ ${WAIT} -ge "${CRITPT_HEALTH_WAIT_SEC}" ]] && { echo "health timeout after ${CRITPT_HEALTH_WAIT_SEC}s"; exit 1; }
  sleep "${CRITPT_HEALTH_POLL_SLEEP}"
  WAIT=$((WAIT + CRITPT_HEALTH_POLL_SLEEP))
done
echo "vLLM ready after ${WAIT}s; resuming OpenDirac CritPt batch..."

if [[ "${CRITPT_RESUME_DRY_RUN:-}" == "1" ]]; then
  echo "CRITPT_RESUME_DRY_RUN=1: skipping CritPt batch (would run run_critpt_open_dirac.py --resume)."
  exit 0
fi

exec uv run --no-sync python scripts/run_critpt_open_dirac.py \
  --resume "${RESUME_DIR}" --timeout "${CRITPT_TIMEOUT}" --concurrency "${CRITPT_CONCURRENCY}"
