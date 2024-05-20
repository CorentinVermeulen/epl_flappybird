#!/bin/bash
#SBATCH --job-name= exp_multi_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --time=05-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -J exp_multi

# Chargement des modules
module load CUDA

# Exécution script Python
python exp_multi.py