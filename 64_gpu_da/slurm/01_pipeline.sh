#!/bin/bash
#
# Job 1: Pipeline (spinup + nature run + free run)
# Runs on 64 GPUs (16 nodes) using MPI
#
#SBATCH --job-name=64gpu-pipe
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=64_gpu_da/slurm/logs/pipeline-%j.out
#SBATCH --error=64_gpu_da/slurm/logs/pipeline-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"
OUTPUT_BASE="${SCRATCH}/DataAssimilocean/64_gpu_da"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

# Enable GPU-aware MPI (required for halo communication with GPU buffers)
export MPICH_GPU_SUPPORT_ENABLED=1

# Disable CUDA.jl stream-ordered memory pool (cudaMallocAsync)
# Pool-allocated memory is not compatible with cuIpcGetMemHandle
# which Cray MPICH uses for GPU-to-GPU communication on the same node
export JULIA_CUDA_MEMORY_POOL=none

echo "=========================================="
echo "64-GPU DA Pipeline (64 GPUs, 1/32 deg)"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${SLURM_NODELIST}"
echo "Tasks:        ${SLURM_NTASKS}"
echo "GPUs:         64 (16 nodes × 4)"
echo "Julia:        $($JULIA --version)"
echo "Output base:  ${OUTPUT_BASE}"
echo "=========================================="

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_BASE}/spinup"
mkdir -p "${OUTPUT_BASE}/nature_run"
mkdir -p "${OUTPUT_BASE}/free_run"

# ===== STEP 1: SPINUP (90 days, 64 GPUs) =====
echo ""
echo "========== STEP 1: SPINUP (90 days, 1/32 deg, 64 GPUs) =========="

# Skip if all 64 rank state files already exist
SPINUP_DONE=true
for r in $(seq 0 63); do
    if [ ! -f "${OUTPUT_BASE}/spinup/final_state_rank${r}.jld2" ]; then
        SPINUP_DONE=false
        break
    fi
done

if [ "$SPINUP_DONE" = true ]; then
    echo "SKIP: Spinup state files already exist, skipping."
else
    echo "Start: $(date)"
    STEP_START=$SECONDS

    srun -n 64 $JULIA --project 64_gpu_da/spinup.jl \
        "${OUTPUT_BASE}/spinup" \
        90

    STEP_EXIT=$?
    STEP_TIME=$(( SECONDS - STEP_START ))
    echo "Spinup finished: exit=${STEP_EXIT}, wall=${STEP_TIME}s ($(( STEP_TIME / 60 ))m)"
    if [ $STEP_EXIT -ne 0 ]; then
        echo "ERROR: Spinup failed!"
        exit $STEP_EXIT
    fi
fi

# ===== STEP 2: NATURE RUN (14 days, 64 GPUs) =====
echo ""
echo "========== STEP 2: NATURE RUN (14 days) =========="

# Skip if all 64 rank surface AND 3D tracer files already exist
NATURE_DONE=true
for r in $(seq 0 63); do
    if [ ! -f "${OUTPUT_BASE}/nature_run/surface_fields_rank${r}.jld2" ] || \
       [ ! -f "${OUTPUT_BASE}/nature_run/tracers_3d_rank${r}.jld2" ]; then
        NATURE_DONE=false
        break
    fi
done

if [ "$NATURE_DONE" = true ]; then
    echo "SKIP: Nature run output files already exist, skipping."
else
    echo "Start: $(date)"
    STEP_START=$SECONDS

    srun -n 64 $JULIA --project 64_gpu_da/nature_run.jl \
        "${OUTPUT_BASE}/spinup" \
        "${OUTPUT_BASE}/nature_run"

    STEP_EXIT=$?
    STEP_TIME=$(( SECONDS - STEP_START ))
    echo "Nature run finished: exit=${STEP_EXIT}, wall=${STEP_TIME}s ($(( STEP_TIME / 60 ))m)"
    if [ $STEP_EXIT -ne 0 ]; then
        echo "ERROR: Nature run failed!"
        exit $STEP_EXIT
    fi
fi

# ===== STEP 3: FREE RUN (14 days, 64 GPUs) =====
echo ""
echo "========== STEP 3: FREE RUN (14 days, perturbed IC) =========="

FREE_DONE=true
for r in $(seq 0 63); do
    if [ ! -f "${OUTPUT_BASE}/free_run/final_state_rank${r}.jld2" ]; then
        FREE_DONE=false
        break
    fi
done

if [ "$FREE_DONE" = true ]; then
    echo "SKIP: Free run state files already exist, skipping."
else
    echo "Start: $(date)"
    STEP_START=$SECONDS

    srun -n 64 $JULIA --project 64_gpu_da/free_run.jl \
        "${OUTPUT_BASE}/spinup" \
        "${OUTPUT_BASE}/free_run"

    STEP_EXIT=$?
    STEP_TIME=$(( SECONDS - STEP_START ))
    echo "Free run finished: exit=${STEP_EXIT}, wall=${STEP_TIME}s ($(( STEP_TIME / 60 ))m)"
    if [ $STEP_EXIT -ne 0 ]; then
        echo "ERROR: Free run failed!"
        exit $STEP_EXIT
    fi
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Total wall time: ${SECONDS}s ($(( SECONDS / 60 ))m)"
echo "Output in: ${OUTPUT_BASE}"
echo "=========================================="
