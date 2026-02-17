#!/bin/bash
#
# Job 2: Ensemble DA (ETKF with 10 members)
#
# Depends on Job 1 completing successfully (spinup + nature run).
#
# Usage:
#   sbatch --dependency=afterok:JOB1_ID single_gpu_da/slurm/02_ensemble_da.sh
#
#SBATCH --job-name=da-etkf
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=single_gpu_da/slurm/logs/etkf-%j.out
#SBATCH --error=single_gpu_da/slurm/logs/etkf-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"
OUTPUT_BASE="${SCRATCH}/DataAssimilocean/single_gpu_da"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

echo "=========================================="
echo "Ensemble DA (ETKF) -- Perlmutter (1 GPU)"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "Julia:        $($JULIA --version)"
echo "Output base:  ${OUTPUT_BASE}"
echo "=========================================="

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_BASE}/da_run"

# Verify prerequisites exist
if [ ! -f "${OUTPUT_BASE}/spinup/final_state.jld2" ]; then
    echo "ERROR: Spinup state not found. Run Job 1 first."
    exit 1
fi

if [ ! -f "${OUTPUT_BASE}/nature_run/surface_fields.jld2" ]; then
    echo "ERROR: Nature run output not found. Run Job 1 first."
    exit 1
fi

echo ""
echo "========== ENSEMBLE DA (10 members, 14 cycles) =========="
echo "Start: $(date)"
DA_START=$SECONDS

$JULIA --project single_gpu_da/ensemble_da.jl \
    "${OUTPUT_BASE}/spinup" \
    "${OUTPUT_BASE}/nature_run" \
    "${OUTPUT_BASE}/free_run" \
    "${OUTPUT_BASE}/da_run"

DA_EXIT=$?
DA_TIME=$(( SECONDS - DA_START ))
echo "DA finished: exit=${DA_EXIT}, wall=${DA_TIME}s ($(( DA_TIME / 60 ))m)"

echo ""
echo "=========================================="
echo "Ensemble DA complete!"
echo "=========================================="
echo "Wall time: ${DA_TIME}s ($(( DA_TIME / 60 ))m)"
echo "Output in: ${OUTPUT_BASE}/da_run"
echo "=========================================="

exit ${DA_EXIT}
