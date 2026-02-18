#!/bin/bash
#
# Job 1: Pipeline (spinup + nature run + free run + comparison)
# Runs on 4 GPUs (1 node) using MPI
#
#SBATCH --job-name=mgda-pipe
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=multi_gpu_da/slurm/logs/pipeline-%j.out
#SBATCH --error=multi_gpu_da/slurm/logs/pipeline-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"
OUTPUT_BASE="${SCRATCH}/DataAssimilocean/multi_gpu_da"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

# Enable GPU-aware MPI (required for halo communication with GPU buffers)
export MPICH_GPU_SUPPORT_ENABLED=1

# Disable CUDA.jl stream-ordered memory pool (cudaMallocAsync)
# Pool-allocated memory is not compatible with cuIpcGetMemHandle
# which Cray MPICH uses for GPU-to-GPU communication on the same node
export JULIA_CUDA_MEMORY_POOL=none

echo "=========================================="
echo "Multi-GPU DA Pipeline (4 GPUs, 1/8 deg)"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${SLURM_NODELIST}"
echo "Tasks:        ${SLURM_NTASKS}"
echo "GPUs:         4"
echo "Julia:        $($JULIA --version)"
echo "Output base:  ${OUTPUT_BASE}"
echo "=========================================="

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_BASE}/spinup"
mkdir -p "${OUTPUT_BASE}/nature_run"
mkdir -p "${OUTPUT_BASE}/free_run"
mkdir -p "${OUTPUT_BASE}/plots"

# ===== STEP 1: SPINUP (90 days, 4 GPUs) =====
echo ""
echo "========== STEP 1: SPINUP (90 days, 1/8 deg, 4 GPUs) =========="

# Skip if all 4 rank state files already exist
SPINUP_DONE=true
for r in 0 1 2 3; do
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

    srun -n 4 $JULIA --project multi_gpu_da/spinup.jl \
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

# ===== STEP 2: NATURE RUN (14 days, 4 GPUs) =====
echo ""
echo "========== STEP 2: NATURE RUN (14 days) =========="

# Skip if all 4 rank surface AND 3D tracer files already exist
NATURE_DONE=true
for r in 0 1 2 3; do
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

    srun -n 4 $JULIA --project multi_gpu_da/nature_run.jl \
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

# ===== STEP 3: FREE RUN (14 days, 4 GPUs) =====
echo ""
echo "========== STEP 3: FREE RUN (14 days, perturbed IC) =========="

FREE_DONE=true
for r in 0 1 2 3; do
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

    srun -n 4 $JULIA --project multi_gpu_da/free_run.jl \
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

# ===== STEP 4: COMPARISON (serial, 1 GPU) =====
echo ""
echo "========== STEP 4: COMPARISON =========="
echo "Start: $(date)"
STEP_START=$SECONDS

$JULIA --project multi_gpu_da/compare_runs.jl \
    "${OUTPUT_BASE}/nature_run" \
    "${OUTPUT_BASE}/free_run" \
    "${OUTPUT_BASE}/plots"

STEP_EXIT=$?
STEP_TIME=$(( SECONDS - STEP_START ))
echo "Comparison finished: exit=${STEP_EXIT}, wall=${STEP_TIME}s"
# Non-fatal if comparison fails

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Total wall time: ${SECONDS}s ($(( SECONDS / 60 ))m)"
echo "Output in: ${OUTPUT_BASE}"
echo "=========================================="
