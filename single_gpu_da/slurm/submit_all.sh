#!/bin/bash
#
# Submit the full DA pipeline with SLURM dependency chaining.
#
# Usage:
#   cd /global/u1/g/glwagner/DataAssimilocean.jl
#   bash single_gpu_da/slurm/submit_all.sh
#

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"
cd "${PROJECT_DIR}"

# Ensure log directory exists
mkdir -p single_gpu_da/slurm/logs

echo "Submitting DA pipeline..."

# Job 1: Spinup + Nature Run + Free Run + Comparison
JOB1=$(sbatch --parsable single_gpu_da/slurm/01_pipeline.sh)
echo "Job 1 (pipeline):    ${JOB1}"

# Job 2: Ensemble DA (depends on Job 1 succeeding)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} single_gpu_da/slurm/02_ensemble_da.sh)
echo "Job 2 (ensemble DA): ${JOB2} (depends on ${JOB1})"

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel ${JOB1} ${JOB2}"
echo ""
echo "Logs will be in: single_gpu_da/slurm/logs/"
echo "Output will be in: \$SCRATCH/DataAssimilocean/single_gpu_da/"
