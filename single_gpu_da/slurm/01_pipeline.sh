#!/bin/bash
#
# Job 1: Spinup + Nature Run + Free Run + Comparison
#
# Runs the full pipeline on a single GPU:
#   1. 90-day spinup at 1/4 degree
#   2. 14-day nature run from spinup end state
#   3. 14-day free run from noised initial condition
#   4. Comparison plots and animations
#
# Usage:
#   sbatch single_gpu_da/slurm/01_pipeline.sh
#
#SBATCH --job-name=da-pipeline
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=single_gpu_da/slurm/logs/pipeline-%j.out
#SBATCH --error=single_gpu_da/slurm/logs/pipeline-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"
OUTPUT_BASE="${SCRATCH}/DataAssimilocean/single_gpu_da"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

echo "=========================================="
echo "DA Pipeline -- Perlmutter (1 GPU)"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "Julia:        $($JULIA --version)"
echo "Output base:  ${OUTPUT_BASE}"
echo "=========================================="

cd "${PROJECT_DIR}"

# Create output directories
mkdir -p "${OUTPUT_BASE}/spinup"
mkdir -p "${OUTPUT_BASE}/nature_run"
mkdir -p "${OUTPUT_BASE}/free_run"
mkdir -p "${OUTPUT_BASE}/plots"

# ============================================================
# Step 1: Spinup (90 days at 1/4 degree)
# ============================================================
echo ""
echo "========== STEP 1: SPINUP (90 days, 1/4 deg) =========="
echo "Start: $(date)"
STEP1_START=$SECONDS

$JULIA --project experiments/stage1_spinup.jl \
    --resolution 0.25 \
    --days 90 \
    --dt 5 \
    --output-dir "${OUTPUT_BASE}/spinup" \
    --no-animation

STEP1_EXIT=$?
STEP1_TIME=$(( SECONDS - STEP1_START ))
echo "Step 1 finished: exit=${STEP1_EXIT}, wall=${STEP1_TIME}s ($(( STEP1_TIME / 60 ))m)"

if [ ${STEP1_EXIT} -ne 0 ]; then
    echo "ERROR: Spinup failed. Aborting pipeline."
    exit ${STEP1_EXIT}
fi

# ============================================================
# Step 2: Nature Run (14 days)
# ============================================================
echo ""
echo "========== STEP 2: NATURE RUN (14 days) =========="
echo "Start: $(date)"
STEP2_START=$SECONDS

$JULIA --project single_gpu_da/nature_run.jl \
    "${OUTPUT_BASE}/spinup" \
    "${OUTPUT_BASE}/nature_run"

STEP2_EXIT=$?
STEP2_TIME=$(( SECONDS - STEP2_START ))
echo "Step 2 finished: exit=${STEP2_EXIT}, wall=${STEP2_TIME}s ($(( STEP2_TIME / 60 ))m)"

if [ ${STEP2_EXIT} -ne 0 ]; then
    echo "ERROR: Nature run failed. Aborting pipeline."
    exit ${STEP2_EXIT}
fi

# ============================================================
# Step 3: Free Run (14 days, noised IC)
# ============================================================
echo ""
echo "========== STEP 3: FREE RUN (14 days, noised IC) =========="
echo "Start: $(date)"
STEP3_START=$SECONDS

$JULIA --project single_gpu_da/free_run.jl \
    "${OUTPUT_BASE}/spinup" \
    "${OUTPUT_BASE}/free_run"

STEP3_EXIT=$?
STEP3_TIME=$(( SECONDS - STEP3_START ))
echo "Step 3 finished: exit=${STEP3_EXIT}, wall=${STEP3_TIME}s ($(( STEP3_TIME / 60 ))m)"

if [ ${STEP3_EXIT} -ne 0 ]; then
    echo "ERROR: Free run failed. Aborting pipeline."
    exit ${STEP3_EXIT}
fi

# ============================================================
# Step 4: Comparison Plots
# ============================================================
echo ""
echo "========== STEP 4: COMPARISON PLOTS =========="
echo "Start: $(date)"
STEP4_START=$SECONDS

$JULIA --project single_gpu_da/compare_runs.jl \
    "${OUTPUT_BASE}/nature_run" \
    "${OUTPUT_BASE}/free_run" \
    "${OUTPUT_BASE}/plots"

STEP4_EXIT=$?
STEP4_TIME=$(( SECONDS - STEP4_START ))
echo "Step 4 finished: exit=${STEP4_EXIT}, wall=${STEP4_TIME}s ($(( STEP4_TIME / 60 ))m)"

# ============================================================
# Summary
# ============================================================
TOTAL_TIME=$SECONDS
echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Step 1 (spinup):     ${STEP1_TIME}s  exit=${STEP1_EXIT}"
echo "Step 2 (nature run): ${STEP2_TIME}s  exit=${STEP2_EXIT}"
echo "Step 3 (free run):   ${STEP3_TIME}s  exit=${STEP3_EXIT}"
echo "Step 4 (comparison): ${STEP4_TIME}s  exit=${STEP4_EXIT}"
echo "Total wall time:     ${TOTAL_TIME}s ($(( TOTAL_TIME / 60 ))m)"
echo "Output in: ${OUTPUT_BASE}"
echo "=========================================="
