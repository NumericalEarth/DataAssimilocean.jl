#!/bin/bash
#
# 8-GPU benchmark: 100 time steps at 1/16 deg for scaling numbers
#
#SBATCH --job-name=8gpu-bench
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=64_gpu_da/slurm/logs/benchmark-8gpu-%j.out
#SBATCH --error=64_gpu_da/slurm/logs/benchmark-8gpu-%j.err

PROJECT_DIR="/global/u1/g/glwagner/DataAssimilocean.jl"

module load julia/1.11.7
JULIA="${JULIA:-julia}"

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none

echo "=========================================="
echo "8-GPU Benchmark (1/16 deg, 100 steps)"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NODELIST}"
echo "Tasks:   ${SLURM_NTASKS}"
echo "=========================================="

cd "${PROJECT_DIR}"
mkdir -p 64_gpu_da/slurm/logs

echo "Start: $(date)"
START=$SECONDS

srun -n 8 $JULIA --project 64_gpu_da/benchmark_8gpu.jl

EXIT_CODE=$?
ELAPSED=$(( SECONDS - START ))
echo "Benchmark finished: exit=${EXIT_CODE}, wall=${ELAPSED}s ($(( ELAPSED / 60 ))m)"
echo "End: $(date)"
