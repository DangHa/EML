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

echo "Running Task 3 Image Augementation: "

echo "Crop and Flip"
python3 exercise04_template.py --epochs 30 --selectaug 1 --output-file "Crop_Flip.json"

echo "Crop, Flip and Rotate"
python3 exercise04_template.py --epochs 30 --selectaug 2 --output-file "Crop_Flip_Rotate.json"

echo "Crop, Flip and Jitter"
python3 exercise04_template.py --epochs 30 --selectaug 3 --output-file "Crop_Flip_Jitter.json"

