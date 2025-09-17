#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --account=pr_119_tandon_priority
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate liger
cd /scratch/zl4789/liger


for dataset_name in Beauty # Toys_and_Games Sports_and_Outdoors
do   
    python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        seed=42 \
        device_id=0 \
        method=base \
        test_method=tiger \
        experiment_id="tiger_$dataset_name"
done

conda deactivate