#!/bin/bash
#SBATCH --job-name=rnspace_byt5
#SBATCH --gres=gpu:a5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=rn3004@princeton.edu

python train.py

# srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py 0 &
# srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py 1 &
# srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py 2 &
# srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py 3 &
# wait