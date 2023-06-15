# moa-profiler
# Repository for paper

## Deep Representation Learning Determines Drug Mechanism of Action from Cell Painting Images 

### Public dataset repos:
JUMP: https://github.com/jump-cellpainting/2021_Chandrasekaran_submitted
LINCS: https://github.com/broadinstitute/lincs-cell-painting

Both datasets are hosted by the Broad Institute on AWS

JUMP1 images:
aws s3 sync --no-sign-request s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/ . 

JUMP single cell data:
aws s3 cp --no-sign-request --recursive s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/backend/2020_11_04_CPJUMP1/ . 

Compressed LINCS png images:
aws s3 sync s3://cellpainting-gallery/cpg0004-lincs/broad/images/2016_04_01_a549_48hr_batch1_compressed/images/ LINCSDatasetCompressed/ --request-payer --profile jump-cp-role

Full LINCS tiff images:
aws s3 sync s3://cellpainting-gallery/cpg0004-lincs/broad/images/ . --request-payer --profile jump-cp-role

LINCS single cell data / locations: 
aws s3 sync s3://cellpainting-gallery/cpg0004-lincs/broad/workspace/backend/ . --request-payer --profile jump-cp-role 

### Channel order of datasets: 
JUMP1 dataset:
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

## Example Scripts
To facilitate ease of use, we've provided example bash scripts to train models (train_job.sh) and evaluate them (parallel_eval.sh)
To run the evaluation script (e.g. in a slurm environment): 
sbatch parallel_eval.sh
