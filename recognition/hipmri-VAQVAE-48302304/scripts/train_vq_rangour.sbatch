#!/bin/bash
#SBATCH -J hipmri_vq
#SBATCH -p comp3710   
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out