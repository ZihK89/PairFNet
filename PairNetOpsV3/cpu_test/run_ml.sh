#!/bin/bash --login
#$ -cwd
#$ -V
#$ -N V3-cpu
#$ -pe smp.pe 4

# load conda environment for relevant python libraries
mamba activate pairnet-gpu

python ~/bin/PairNetOpsV3/network.py > out.log
# close conda environment
mamba deactivate
