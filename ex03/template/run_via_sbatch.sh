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
# echo "CPU runing"
# python3 exercise03_template.py --dataset mnist --model mlp --epochs 30 --no-cuda --output-file "task01_cpu.json"
echo "GPU runing"
python3 exercise03_template.py --dataset svhn --model cnn --lr 0.001 --epochs 30 --output-file "task01_vgg.json"

# echo "Running Task 2: "
# echo "MLP runing"
# python3 exercise03_template.py --dataset svhn --model mlp --epochs 30 --output-file "task02_mlp.json"
# echo "CNN runing"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.01 --epochs 30 --output-file "task02_cnn.json"

# echo Training with SGD optimizer
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.1 --epochs 30 --optimizer sgd --output-file "task03_sgd_0.1.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.05 --epochs 30 --optimizer sgd --output-file "task03_sgd_0.05.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.01 --epochs 30 --optimizer sgd --output-file "task03_sgd_0.01.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.005 --epochs 30 --optimizer sgd --output-file "task03_sgd_0.005.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.001 --epochs 30 --optimizer sgd --output-file "task03_sgd_0.001.json"

# echo Training with ADAM optimizer
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.1 --epochs 30 --optimizer adam --output-file "task03_adam_0.1.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.05 --epochs 30 --optimizer adam --output-file "task03_adam_0.05.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.01 --epochs 30 --optimizer adam --output-file "task03_adam_0.01.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.005 --epochs 30 --optimizer adam --output-file "task03_adam_0.005.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.001 --epochs 30 --optimizer adam --output-file "task03_adam_0.001.json"

# echo Training with ADAMAX optimizer
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.1 --epochs 30 --optimizer adamax --output-file "task03_adamax_0.1.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.05 --epochs 30 --optimizer adamax --output-file "task03_adamax_0.05.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.01 --epochs 30 --optimizer adamax --output-file "task03_adamax_0.01.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.005 --epochs 30 --optimizer adamax --output-file "task03_adamax_0.005.json"
# python3 exercise03_template.py --dataset svhn --model cnn --lr 0.001 --epochs 30 --optimizer adamax --output-file "task03_adamax_0.001.json"