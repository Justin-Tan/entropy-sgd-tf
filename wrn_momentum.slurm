#!/bin/bash
#SBATCH --partition physics-gpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --gres gpu:2
#SBATCH --time 96:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name wrn_28_10
#SBATCH --output wrn_momentum-%J.log

# Load modules
module load CUDA/8.0.44
export LD_LIBRARY_PATH=/data/projects/punim0011/cuda/lib64:$LD_LIBRARY_PATH

python3 train.py -opt momentum -n wrn_momentum_28_10
