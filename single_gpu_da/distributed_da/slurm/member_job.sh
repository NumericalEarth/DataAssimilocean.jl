#!/bin/bash
#
# Ensemble member forecast: run a batch of members for one DA cycle
#
# Required environment variables (set via sbatch --export):
#   CYCLE        - DA cycle number (1-based)
#   MEMBER_START - First member in this batch
#   MEMBER_END   - Last member in this batch
#   N_CYCLES     - Total number of DA cycles (optional, default 14)
#
#SBATCH --job-name=da-member
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=single_gpu_da/distributed_da/slurm/logs/%x-%j.out
#SBATCH --error=single_gpu_da/distributed_da/slurm/logs/%x-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"
OUTPUT_BASE="${SCRATCH}/DataAssimilocean/single_gpu_da"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

echo "=========================================="
echo "Distributed DA: Member Forecast"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "Cycle:         ${CYCLE}"
echo "Members:       ${MEMBER_START} to ${MEMBER_END}"
echo "N_CYCLES:      ${N_CYCLES:-14}"
echo "=========================================="

cd "${PROJECT_DIR}"

SPINUP_DIR="${OUTPUT_BASE}/spinup"
DA_DIR="${OUTPUT_BASE}/distributed_da"

# Verify inputs
if [ ! -f "${SPINUP_DIR}/final_state.jld2" ]; then
    echo "ERROR: Spinup state not found"
    exit 1
fi

echo "Start: $(date)"
START=$SECONDS

$JULIA --project single_gpu_da/distributed_da/ensemble_member.jl \
    "${SPINUP_DIR}" \
    "${DA_DIR}" \
    "${DA_DIR}" \
    "${CYCLE}" \
    "${MEMBER_START}" \
    "${MEMBER_END}" \
    "${N_CYCLES:-14}"

EXIT_CODE=$?
ELAPSED=$(( SECONDS - START ))
echo "Member forecast finished: exit=${EXIT_CODE}, wall=${ELAPSED}s ($(( ELAPSED / 60 ))m)"

exit ${EXIT_CODE}
