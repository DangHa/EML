#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 2
#SBATCH --time 00:30:00
#SBATCH -p exercise
#SBATCH -o slurm_output.log

echo "Loading conda env"
module load anaconda/3
source ~/.bashrc
conda activate EML_ex03

echo "Running Task 1 Dropout: "

echo "Dropout 0.1"
python3 exercise04_template.py --epochs 30 --dropout_p 0.1 --output-file "task01_dropout01.json"

echo "Dropout 0.4"
python3 exercise04_template.py --epochs 30 --dropout_p 0.4 --output-file "task01_dropout04.json"

echo "Dropout 0.7"
python3 exercise04_template.py --epochs 30 --dropout_p 0.7 --output-file "task01_dropout07.json"
