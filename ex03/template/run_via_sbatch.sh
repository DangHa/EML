#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 30:00
#SBATCH -p exercise
#SBATCH -o slurm_output.log

echo "Loading conda env"
module load anaconda/3
source ~/.bashrc
conda activate EML_ex03


echo "Running Task 1: "
echo "CPU runing"
python3 template/exercise03_template.py --dataset mnist --model mlp --epochs 30 --no-cuda --output-file "task01_cpu.json"
echo "GPU runing"
python3 template/exercise03_template.py --dataset mnist --model mlp --epochs 30 --output-file "task01_gpu.json"

echo "Running Task 2: "
echo "MLP runing"
python3 template/exercise03_template.py --dataset svhn --model mlp --epochs 30 --output-file "task02_mlp.json"
echo "CNN runing"
python3 template/exercise03_template.py --dataset svhn --model cnn --lr 0.01 --epochs 30 --output-file "task02_cnn.json"