#!/bin/bash
#SBATCH --job-name=glider_collection
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --account=aip-rrabba
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


source .env
module load java/21.0.1
deepspeed --num_gpus 4 glider_data_collection.py