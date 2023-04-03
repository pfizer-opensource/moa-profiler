#!/bin/bash
#SBATCH -o logs/training_log.txt
#SBATCH -J MP-train 
#SBATCH --time=21-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --qos=interactive

echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo " Run started at:- "
date
source ~/.bashrc
conda activate env2 
## standard training on JUMP1 dataset
python classification_runner.py --distributed_data_parallel --mode=train --study=JUMP1 --label_type=moa_targets_compounds --n_epochs=100 --eval_batch_size=14 --num_training_gpus=4 --print_freq=100
##standard training on LINCS dataset 
python classification_runner.py --distributed_data_parallel --mode=train --study=lincs --label_type=moas_10uM --n_epochs=100 --eval_batch_size=14 --num_training_gpus=4 --print_freq=100
##four channel LINCS model 
python classification_runner.py --distributed_data_parallel --mode=train --study=lincs --label_type=moas_10uM_four_channel --n_epochs=100 --eval_batch_size=14 --num_training_gpus=4 --print_freq=100 --in_channels=4
##four channel JUMP1 model 
python classification_runner.py --distributed_data_parallel --mode=train --study=JUMP1 --label_type=moa_targets_compounds_four_channel --n_epochs=100 --eval_batch_size=14 --num_training_gpus=4 --print_freq=100 --in_channels=4
##replicate limit LINCS
for REPLICATE_NUMBER in 1 2 3 4 5 6 7
do 
    python classification_runner.py --distributed_data_parallel --mode=train --study=lincs --label_type=moas_10uM_replicates=$REPLICATE_NUMBER --n_epochs=100 --eval_batch_size=14 --num_training_gpus=4 --print_freq=100
done
##replicate limit JUMP1
for REPLICATE_NUMBER in 4 8 12 16 20 24 28 32
do 
    python classification_runner.py --distributed_data_parallel --mode=train --study=JUMP1 --label_type=moa_targets_compounds_replicates=$REPLICATE_NUMBER --n_epochs=100 --eval_batch_size=5 --num_training_gpus=4 --print_freq=100
done
echo "Run completed at:- "
date
