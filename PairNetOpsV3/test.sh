#!/bin/bash --login
#$ -cwd
#$ -V
#$ -N V3-gpu
#$ -l v100
#$ -pe smp.pe 4

# load conda environment for relevant python libraries
mamba activate pairnet-gpu
module load apps/binapps/pytorch/2.3.0-311-gpu-cu121
python ~/bin/PairNetOpsV3/network.py > out.log
# close conda environment
mamba deactivate