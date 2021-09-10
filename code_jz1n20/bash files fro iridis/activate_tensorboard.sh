#!/bin/bash

##This file sets up the SSH tunnel so you can view tensorboard stats from iridis nodes
# usage: run with ./activate_tensorboard.sh [ENVIRONMENT_NAME] [MODEL_NAME]
# example: ./activate_tensorboard.sh env3 Model_KS_02
#
# remember to chmod 777 activate_tensorboard.sh if you get a permissions error.
#
# Modifying this file: make sure your line endings are unix based before you push. (bottom right of PyCharm)
module load conda 
# On the iridis node run:
source activate DeepL
python -m tensorboard.main --logdir = "/mainfs/scratch/jz1n20/SE_Resnet18/$2/TensorLogs/"--port 6006

# In your local terminal run:
# ssh -N -f -L 6006:127.0.0.1:6006 jz1n20@iridis5_a.soton.ac.uk

# you can end up with multiple ssh -N -f -l commands running on your PC, to get rid of them, you could restart your PC
# or you could type:
# ps aux | grep ssh
# find the ssh line, and there should be a number in the left hand column which identifies that process (4/5 digits long).
# kill the process with:
# kill [NUMBER_OF_PROCESS]