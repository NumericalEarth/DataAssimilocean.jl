#!/bin/bash
#
# 64-GPU benchmark: 1000 time steps at 1/32 deg for scaling numbers
#
#SBATCH --job-name=64gpu-bench
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=64_gpu_da/slurm/logs/benchmark-%j.out
#SBATCH --error=64_gpu_da/slurm/logs/benchmark-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none

echo "=========================================="
echo "64-GPU Benchmark (1/32 deg, 1000 steps)"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NODELIST}"
echo "Tasks:   ${SLURM_NTASKS}"
echo "=========================================="

cd "${PROJECT_DIR}"
mkdir -p 64_gpu_da/slurm/logs

echo "Start: $(date)"
START=$SECONDS

srun -n 64 $JULIA --project 64_gpu_da/benchmark.jl

EXIT_CODE=$?
ELAPSED=$(( SECONDS - START ))
echo "Benchmark finished: exit=${EXIT_CODE}, wall=${ELAPSED}s ($(( ELAPSED / 60 ))m)"
echo "End: $(date)"
