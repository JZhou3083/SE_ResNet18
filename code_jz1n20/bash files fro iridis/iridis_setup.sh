#!/bin/bash

###setup conda environment
#
#you'll need to do these command first before you do anything with conda:
#mkdir /scratch/ch1u20/.conda
#ln -s /scratch/ch1u20/.conda ~/.conda
#
#
#Then, alter the permissions with: chmod u+x iridis_setup.sh
#and run it with: source iridis_setup.sh
#
# Modification: make sure your line endings are unix based before you push. (bottom right of PyCharm)
#
# Other Info:
# delete bad environments with "conda remove --name env1 --all"
# to find out what packages you have in your environment: "conda list"

##Load conda
module load conda/py3-latest

##Create Environment
yes | conda remove --name DeepL --all
conda config --add channels anaconda
conda config --add channels conda-forge
yes | conda create -n DeepL python=3.8 
source activate DeepL
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install torchsummary
pip install numpy
pip install torchbearer
pip install scikit-learn
pip install 'git+https://github.com/lanpa/tensorboardX'
pip install scikit-learn
pip install matplotlib
conda deactivate