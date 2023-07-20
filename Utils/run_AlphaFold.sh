#!/bin/bash

#This script was used to run AlphaFold on the PALMA-II HPC cluster (https://confluence.uni-muenster.de/display/HPC/High+Performance+Computing)


#SBATCH --nodes=1
#SBATCH --ntasks 18
#SBATCH --cpus-per-task 6
#SBATCH --mem=80G
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:3
#SBATCH --array=1-2000


module --force purge

#load required modules of HPC cluster
ml palma/2021a
ml GCC/10.3.0
ml OpenMPI/4.1.1
ml AlphaFold/2.1.1-CUDA-11.3.1
wait
export ALPHAFOLD_DATA_DIR=/Applic.HPC/data/alphafold

cd $WorkingDir

alphafold \
    --fasta_paths=$FASTAPATH/$ID${SLURM_ARRAY_TASK_ID}.fa \
    --model_preset=monomer \
    --output_dir=$ReslutsDir/${SLUMR_ARRAY_ID} \
    --max_template_date=2022-12-19 \
    --is_prokaryote_list=false \
    --db_preset=reduced_dbs \
    --data_dir=/Applic.HPC/data/alphafold \

#ENV
