#!/bin/bash
###Run on iridis
#
# Before you run this, make sure your environment is set up correctly (env1).
# To run:
# sbatch run_on_iridis.sh
#
# Modification: make sure your line endings are unix based before you push. (bottom right of PyCharm)
#
# Other Info:
# squeue -lu [username] to keep track of your job
# scancel [jobID] to cancel your job
# ssh [NodeName] to ssh onto the node where your job is running

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --ntasks=10

module load cuda/10.2

module load conda
source activate ~/.conda/envs/DeepL
conda info --envs


python se_resnet18.py
