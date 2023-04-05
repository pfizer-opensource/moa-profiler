# moa-profiler

## Deep Representation Learning Determines Drug Mechanism of Action from Cell Painting Images 

### Public dataset repos:
JUMP1 Pilot: https://github.com/jump-cellpainting/2021_Chandrasekaran_submitted
LINCS: https://github.com/broadinstitute/lincs-cell-painting

Both datasets are hosted by the Broad Institute on AWS and are publicly available

JUMP1 Pilot images:
aws s3 sync --no-sign-request s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/ . 

JUMP single cell data:
aws s3 cp --no-sign-request --recursive s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/backend/2020_11_04_CPJUMP1/ . 

LINCS compressed png images: 
aws s3 sync s3://jump-cellpainting/projects/2015_10_05_DrugRepurposing_AravindSubramanian_GolubLab_Broad/2016_04_01_a549_48hr_batch1_compressed/images/ . --request-payer --profile jump-cp-role

Full tiff images:
aws s3 sync s3://cellpainting-gallery/lincs/broad/images/2016_04_01_a549_48hr_batch1/images/ . --request-payer --profile jump-cp-role

lincs single cell data / locations: 
aws s3 sync s3://cellpainting-gallery/cpg0004-lincs/broad/workspace/backend/ . --request-payer --profile jump-cp-role 

place all data in a folder called "data/"

### Channel order of datasets: 
JUMP1 Pilot dataset:
    ch01 - Alexa 647 Mito
    ch02 - Alexa 568 AGP (actin cytoskeleton, golgi, plasma membrane)
    ch03 - Alexa 488 long RNA
    ch04 - Alexa 488 ER
    ch05 - Hoechst 33342 DNA

LINCS dataset: 
    ch01 - HOECHST 33342 DNA
    ch02 - Alexa 488 ER 
    ch03 - 488 long RNA
    ch04 - Alexa 568 AGP
    ch05 - Alexa 647 Mito

### Software environment 
A conda .yml environment is provided called env2.yml 
To create a conda environment called env2: 
conda env create -f env2.yml -n env2
We used PyTorch 1.9.0 with CUDA 11.1, which can be installed by: 
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

## Example scripts
To facilitate ease of use, we've provided example bash scripts to train models (train_job.sh) and evaluate them (parallel_eval.sh)
To run the evaluation script (e.g. in a slurm environment): 
sbatch parallel_eval.sh

figures are written to a directory called "outputs/" so be sure to create this directory first

## Fully trained models
Models trained on the JUMP1 Pilot dataset:
    well holdout: save_dir/JUMP1/multiclass_classification/Jun-22-2022-08:49:11/models/model_best.dat
    compound holdout: save_dir/JUMP1/multiclass_classification/Oct-18-2022-18:07:09/models/model_best.dat

Models trained on the LINCS dataset: 
    well holdout: save_dir/lincs/multiclass_classification/Jul-04-2022-15:00:03/models/model_best.dat
    compound holdout: save_dir/lincs/multiclass_classification/Oct-23-2022-16:27:06/models/model_best.dat


