#!/bin/bash
#
# Job 2: Ensemble DA (ETKF with 10 members, 4 GPUs)
#
# Depends on Job 1 completing (spinup + nature run).
#
# Usage:
#   sbatch --dependency=afterok:JOB1_ID multi_gpu_da/slurm/02_ensemble_da.sh
#
#SBATCH --job-name=mgda-etkf
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=multi_gpu_da/slurm/logs/etkf-%j.out
#SBATCH --error=multi_gpu_da/slurm/logs/etkf-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"
OUTPUT_BASE="${SCRATCH}/DataAssimilocean/multi_gpu_da"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

# Enable GPU-aware MPI (required for halo communication with GPU buffers)
export MPICH_GPU_SUPPORT_ENABLED=1

# Disable CUDA.jl stream-ordered memory pool (cudaMallocAsync)
# Pool-allocated memory is not compatible with cuIpcGetMemHandle
export JULIA_CUDA_MEMORY_POOL=none

echo "=========================================="
echo "Ensemble DA (ETKF) -- 4 GPUs, 1/8 deg"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${SLURM_NODELIST}"
echo "Julia:        $($JULIA --version)"
echo "=========================================="

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_BASE}/da_run"

# Verify prerequisites
for r in 0 1 2 3; do
    if [ ! -f "${OUTPUT_BASE}/spinup/final_state_rank${r}.jld2" ]; then
        echo "ERROR: Spinup state rank${r} not found."
        exit 1
    fi
done

if [ ! -f "${OUTPUT_BASE}/nature_run/surface_fields_rank0.jld2" ]; then
    echo "ERROR: Nature run surface fields not found."
    exit 1
fi

echo ""
echo "========== ENSEMBLE DA (10 members, 14 cycles, 4 GPUs) =========="
echo "Start: $(date)"
DA_START=$SECONDS

srun -n 4 $JULIA --project multi_gpu_da/ensemble_da.jl \
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
echo "Wall time: ${DA_TIME}s ($(( DA_TIME / 60 ))m)"
echo "Output in: ${OUTPUT_BASE}/da_run"
echo "=========================================="

exit ${DA_EXIT}
