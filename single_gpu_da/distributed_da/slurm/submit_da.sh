#!/bin/bash
#
# Master orchestration script: submit all DA jobs with SLURM dependency chains
#
# Submits all jobs upfront — no monitoring needed. SLURM handles ordering.
#
# Usage:
#   bash single_gpu_da/distributed_da/slurm/submit_da.sh [N_CYCLES] [N_ENS] [MEMBERS_PER_JOB]
#
# Example (small test):
#   bash single_gpu_da/distributed_da/slurm/submit_da.sh 2 10 4
#
# Example (full run):
#   bash single_gpu_da/distributed_da/slurm/submit_da.sh 14 10 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_CYCLES=${1:-14}
N_ENS=${2:-10}
MEMBERS_PER_JOB=${3:-4}

echo "=========================================="
echo "Distributed DA: Submitting Job Pipeline"
echo "=========================================="
echo "  Cycles:          ${N_CYCLES}"
echo "  Ensemble size:   ${N_ENS}"
echo "  Members per job: ${MEMBERS_PER_JOB}"

# Compute number of member jobs per cycle
N_MEMBER_JOBS=$(( (N_ENS + MEMBERS_PER_JOB - 1) / MEMBERS_PER_JOB ))
TOTAL_JOBS=$(( 1 + N_CYCLES * (N_MEMBER_JOBS + 1) ))

echo "  Member jobs/cycle: ${N_MEMBER_JOBS}"
echo "  Total SLURM jobs:  ${TOTAL_JOBS} (1 init + ${N_CYCLES} x $(( N_MEMBER_JOBS + 1 )))"
echo "=========================================="

# Create log directory
mkdir -p "${SCRIPT_DIR}/logs"

# --- Step 1: Submit initialization job ---
echo ""
echo "Submitting init job..."
INIT_JOB=$(sbatch --parsable \
    --export=ALL,N_ENS=${N_ENS} \
    "${SCRIPT_DIR}/init_job.sh")
echo "  Init job: ${INIT_JOB}"

PREV_DEPENDENCY="${INIT_JOB}"

# --- Step 2: For each cycle, submit member jobs + analysis job ---
for cycle in $(seq 1 ${N_CYCLES}); do
    echo ""
    echo "--- Cycle ${cycle}/${N_CYCLES} ---"

    MEMBER_JOBS=""

    # Submit member batch jobs
    for batch in $(seq 1 ${N_MEMBER_JOBS}); do
        MEMBER_START=$(( (batch - 1) * MEMBERS_PER_JOB + 1 ))
        MEMBER_END=$(( batch * MEMBERS_PER_JOB ))
        if [ ${MEMBER_END} -gt ${N_ENS} ]; then
            MEMBER_END=${N_ENS}
        fi

        JOB_NAME="c${cycle}-m${MEMBER_START}-${MEMBER_END}"
        MEMBER_JOB=$(sbatch --parsable \
            --dependency=afterok:${PREV_DEPENDENCY} \
            --job-name="${JOB_NAME}" \
            --export=ALL,CYCLE=${cycle},MEMBER_START=${MEMBER_START},MEMBER_END=${MEMBER_END},N_CYCLES=${N_CYCLES} \
            "${SCRIPT_DIR}/member_job.sh")

        echo "  Member job ${JOB_NAME}: ${MEMBER_JOB} (depends on ${PREV_DEPENDENCY})"

        if [ -z "${MEMBER_JOBS}" ]; then
            MEMBER_JOBS="${MEMBER_JOB}"
        else
            MEMBER_JOBS="${MEMBER_JOBS}:${MEMBER_JOB}"
        fi
    done

    # Submit analysis job (depends on ALL member jobs for this cycle)
    ANALYSIS_JOB_NAME="c${cycle}-analysis"
    ANALYSIS_JOB=$(sbatch --parsable \
        --dependency=afterok:${MEMBER_JOBS} \
        --job-name="${ANALYSIS_JOB_NAME}" \
        --export=ALL,CYCLE=${cycle},N_CYCLES=${N_CYCLES},N_ENS=${N_ENS} \
        "${SCRIPT_DIR}/analysis_job.sh")

    echo "  Analysis job ${ANALYSIS_JOB_NAME}: ${ANALYSIS_JOB} (depends on ${MEMBER_JOBS})"

    # Next cycle's member jobs depend on this cycle's analysis
    PREV_DEPENDENCY="${ANALYSIS_JOB}"
done

echo ""
echo "=========================================="
echo "All ${TOTAL_JOBS} jobs submitted!"
echo "=========================================="
echo "Monitor with: squeue -u \$USER --format='%.10i %.20j %.8T %.10M %.12l %.6D %R'"
echo "Cancel all:   scancel --dependency=afterok:${INIT_JOB}"
echo "=========================================="
