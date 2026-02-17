#!/bin/bash
#
# Debug version: ETKF with only 3 cycles (fits in 30-min debug queue)
#
#SBATCH --job-name=da-debug
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=single_gpu_da/slurm/logs/etkf-debug-%j.out
#SBATCH --error=single_gpu_da/slurm/logs/etkf-debug-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"
OUTPUT_BASE="${SCRATCH}/DataAssimilocean/single_gpu_da"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

echo "=========================================="
echo "ETKF DEBUG RUN (3 cycles only)"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "=========================================="

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_BASE}/da_debug"

$JULIA --project single_gpu_da/ensemble_da.jl \
    "${OUTPUT_BASE}/spinup" \
    "${OUTPUT_BASE}/nature_run" \
    "${OUTPUT_BASE}/free_run" \
    "${OUTPUT_BASE}/da_debug" \
    3

DA_EXIT=$?
echo "Debug DA finished: exit=${DA_EXIT}"
exit ${DA_EXIT}
