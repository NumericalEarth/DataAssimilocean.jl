#!/bin/bash
#
# Submit the full multi-GPU DA pipeline with dependency chaining
#
# Usage:
#   bash multi_gpu_da/slurm/submit_all.sh

cd /global/u1/g/glwagner/DataAssimilocean.jl

# Ensure log directory exists
mkdir -p multi_gpu_da/slurm/logs

echo "Submitting Multi-GPU DA Pipeline (4 GPUs, 1/8 degree)..."

# Job 1: Pipeline (spinup + nature + free + compare)
JOB1=$(sbatch --parsable multi_gpu_da/slurm/01_pipeline.sh)
echo "Job 1 (pipeline):     ${JOB1}"

# Job 2: Ensemble DA (depends on Job 1)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} multi_gpu_da/slurm/02_ensemble_da.sh)
echo "Job 2 (ensemble DA):  ${JOB2} (depends on ${JOB1})"

echo ""
echo "Monitor: squeue -u $(whoami)"
echo "Logs:    multi_gpu_da/slurm/logs/"
