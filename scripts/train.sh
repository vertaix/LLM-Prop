#!/bin/bash
#SBATCH --job-name=rnspace_byt5
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=rn3004@princeton.edu

python /train.py

cat slurm-*.out | more
