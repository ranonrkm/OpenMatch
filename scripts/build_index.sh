#!/bin/bash
#SBATCH --no-requeue
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=dataset
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=64G
#SBATCH --output=logs/index_%j.out
#SBATCH --error=logs/index_%j.err

srun torchrun --standalone --nproc_per_node 4 -m openmatch.driver.build_index  --output_dir /data/user_data/rsadhukh/wikipedia/contriever \
 --model_name_or_path facebook/contriever --cache_dir /data/user_data/rsadhukh/cache \
 --dataset wikipedia --version 20220301.en \
 --per_device_eval_batch_size 256   --p_max_len 512  --dataloader_num_workers 1  --pooling mean  --doc_template "<Clean-Text>"