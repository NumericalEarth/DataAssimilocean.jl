#!/bin/bash
#
# ETKF analysis: load forecasts, compute weights, apply update
#
# Required environment variables (set via sbatch --export):
#   CYCLE    - DA cycle number (1-based)
#   N_CYCLES - Total number of DA cycles (optional, default 14)
#   N_ENS    - Ensemble size (optional, default 10)
#
#SBATCH --job-name=da-analysis
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
echo "Distributed DA: ETKF Analysis"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Cycle:     ${CYCLE}"
echo "N_CYCLES:  ${N_CYCLES:-14}"
echo "N_ENS:     ${N_ENS:-10}"
echo "=========================================="

cd "${PROJECT_DIR}"

SPINUP_DIR="${OUTPUT_BASE}/spinup"
NATURE_DIR="${OUTPUT_BASE}/nature_run"
FREE_DIR="${OUTPUT_BASE}/free_run"
DA_DIR="${OUTPUT_BASE}/distributed_da"
DIAG_DIR="${DA_DIR}/diagnostics"
mkdir -p "${DIAG_DIR}"

# Verify inputs
if [ ! -f "${SPINUP_DIR}/final_state.jld2" ]; then
    echo "ERROR: Spinup state not found"
    exit 1
fi

if [ ! -f "${NATURE_DIR}/tracers_3d.jld2" ]; then
    echo "ERROR: Nature run 3D tracers not found"
    exit 1
fi

echo "Start: $(date)"
START=$SECONDS

$JULIA --project single_gpu_da/distributed_da/etkf_analysis.jl \
    "${SPINUP_DIR}" \
    "${NATURE_DIR}" \
    "${DA_DIR}" \
    "${DA_DIR}" \
    "${CYCLE}" \
    "${FREE_DIR}" \
    "${DIAG_DIR}" \
    "${N_CYCLES:-14}" \
    "${N_ENS:-10}"

EXIT_CODE=$?
ELAPSED=$(( SECONDS - START ))
echo "Analysis finished: exit=${EXIT_CODE}, wall=${ELAPSED}s ($(( ELAPSED / 60 ))m)"

exit ${EXIT_CODE}
