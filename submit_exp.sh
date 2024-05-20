#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -J exp_multi

# Chargement des modules
# module load python/3.x

# Exécution script Python
python exp_multi.py