#!/bin/bash
#SBATCH --partition gpgpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --gres gpu:2
#SBATCH -A punim0514
#SBATCH -p gpgpu
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name wrn_gpgpu
#SBATCH --output wrn-%J.log

# Load modules
# module load CUDA/8.0.44
# export LD_LIBRARY_PATH=/data/projects/punim0011/cuda/lib64:$LD_LIBRARY_PATH
# module load CUDA/9.2.88-intel-2017.u2

date

python3 train.py -opt adam -n mlhep -d cifar10
