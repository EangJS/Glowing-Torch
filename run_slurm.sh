#!/bin/bash
#SBATCH --partition=medium
#SBATCH --gpus=titanv:1  # Request 1 v100 GPU
#SBATCH --nodes=1  # Request 1 node
#SBATCH --time=01:00:00  # Expected time for job completion
srun python classification/BERT/set-fit.py

