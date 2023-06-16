#!/bin/bash
#SBATCH -o logs/name_of_your_log.txt
#SBATCH -J MPPublic
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=300G 
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:1
#SBATCH --nodes=2

echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo " Run started at:- "
date
source ~/.bashrc
conda activate env2 

cd /home/wongd26/workspace/profiler/public/moa-profiler/

metric="pearson"
standardization_type="by_DMSO_plate"

# # #evaluate JUMP1 moa_targets_compounds polycompound only 
echo "evaluating JUMP1 efficientnet moa_targets_compounds_polycompound"
srun --ntasks=1 --nodes=1 --gres=gpu:1 python classification_runner.py --mode=eval --distributed_data_parallel --study=JUMP1 --label_type=moa_targets_compounds_polycompound --class_aggregator=median --well_aggregator=median --metric=$metric --standardization_type=$standardization_type &
# # # ##evaluate lincs moas_10uM polycompound only
echo "evaluating lincs efficientnet moas_10uM_polycompound"
srun --ntasks=1 --nodes=1 --gres=gpu:1 python classification_runner.py --mode=eval --distributed_data_parallel --study=lincs --label_type=moas_10uM_polycompound --class_aggregator=median --well_aggregator=median --metric=$metric --standardization_type=$standardization_type &
wait 

##eval compound holdouts
echo "evaluating JUMP1 efficientnet moa_targets_compounds_holdout_2"
srun --ntasks=1 --nodes=1 --gres=gpu:1 python classification_runner.py --mode=eval_compound_holdout --distributed_data_parallel --study=JUMP1 --label_type=moa_targets_compounds_holdout_2 --class_aggregator=median --well_aggregator=median --metric=$metric --standardization_type=$standardization_type &
echo "evaluating lincs efficientnet moas_10uM_compounds_holdout_2"
srun --ntasks=1 --nodes=1 --gres=gpu:1 python classification_runner.py --mode=eval_compound_holdout --distributed_data_parallel --study=lincs --label_type=moas_10uM_compounds_holdout_2 --class_aggregator=median --well_aggregator=median --metric=$metric --standardization_type=$standardization_type &
wait
python plot.py --class_aggregator=median --well_aggregator=median --metric=$metric
wait
echo "Run completed at:- "
date