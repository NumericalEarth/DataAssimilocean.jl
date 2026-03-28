#!/bin/bash
#
# Initialize ensemble: create perturbed member states from spinup
#
#SBATCH --job-name=da-init
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=single_gpu_da/distributed_da/slurm/logs/init-%j.out
#SBATCH --error=single_gpu_da/distributed_da/slurm/logs/init-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"
OUTPUT_BASE="${SCRATCH}/DataAssimilocean/single_gpu_da"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

echo "=========================================="
echo "Distributed DA: Initialize Ensemble"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "N_ENS:   ${N_ENS:-10}"
echo "=========================================="

cd "${PROJECT_DIR}"

SPINUP_DIR="${OUTPUT_BASE}/spinup"
DA_DIR="${OUTPUT_BASE}/distributed_da"
mkdir -p "${DA_DIR}"

# Verify spinup exists
if [ ! -f "${SPINUP_DIR}/final_state.jld2" ]; then
    echo "ERROR: Spinup state not found at ${SPINUP_DIR}/final_state.jld2"
    exit 1
fi

echo "Start: $(date)"
START=$SECONDS

$JULIA --project single_gpu_da/distributed_da/initialize_ensemble.jl \
    "${SPINUP_DIR}" \
    "${DA_DIR}" \
    "${N_ENS:-10}"

EXIT_CODE=$?
ELAPSED=$(( SECONDS - START ))
echo "Init finished: exit=${EXIT_CODE}, wall=${ELAPSED}s ($(( ELAPSED / 60 ))m)"

exit ${EXIT_CODE}
