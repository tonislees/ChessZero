#!/bin/bash
#SBATCH -J ml_training
#SBATCH --partition=gpu
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load python/3.12.3

cd ChessZero
source venv/bin/activate

python /src/train.py