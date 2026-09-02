#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --account=turnelln-cluster-challenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:05:00
#SBATCH --out=out/nccl-%j.out

module purge
module load bluebear
module load bear-apps/2024a/live
module load NCCL/2.26.2-GCCcore-13.3.0-CUDA-12.6.0


./build/all_reduce_perf -b 1G -e 1G -g 2
