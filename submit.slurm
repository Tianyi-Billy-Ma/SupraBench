#!/bin/bash
#SBATCH --job-name=dynamic_exp_0
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=bglo-delta-gpu
#SBATCH -o logs/slurm-%j.out
#SBATCH --no-requeue
#SBATCH --mail-user=wsun4+code@nd.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module load cuda/12.8

/work/nvme/bglo/wsun4/SupraBench
source .venv/bin/activate

python3 src/vqa/run_identification.py --mode smiles --model /work/nvme/bglo/wsun4/SupraBench/models/Qwen3.5-9B

python3 src/vqa/run_identification.py --mode smiles --model /work/nvme/bglo/wsun4/SupraBench/models/Qwen3.5-27B