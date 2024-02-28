#!/bin/bash
#SBATCH --partition=medium
#SBATCH --gpus=v100:1  # Request 1 v100 GPU
#SBATCH --nodes=1  # Request 1 node
#SBATCH --time=03:00:00  # Expected time for job completion
#SBATCH --job-name="bleu"
srun python NativeBERTFashion.py

