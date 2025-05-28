#!/bin/bash
 
#SBATCH --job-name=rnn_training            
#SBATCH --account=st-alexrod6-1-gpu    
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12                           
#SBATCH --mem=128G                  
#SBATCH --time=12:00:00             
#SBATCH --gpus=4
#SBATCH --output=output.txt         
#SBATCH --error=error.txt          
#SBATCH --mail-user=<your-email> # <--- change to your email
#SBATCH --mail-type=ALL    
#SBATCH --constraint=gpu_mem_32                         
 
module load gcc
module load cuda
module load miniconda3

eval "$(conda shell.bash hook)"
conda activate mds-afforest-dev
 
# Define a timestamped scratch run directory
TS=$(date +%s)
SCRATCH_RUN_DIR=/scratch/st-alexrod6-1-gpu/$USER/afforest_run_$TS

# Copy your GitHub repo to scratch (excluding .git)
rsync -a --exclude='.git' /arc/home/$USER/MDSAfforestationMonitoring "$SCRATCH_RUN_DIR"

# Move to the scratch working directory
cd "$SCRATCH_RUN_DIR"

# Make sure the logs directory exists for SLURM output files
mkdir -p logs

# go to root of repo
cd "$SCRATCH_RUN_DIR/MDSAfforestationMonitoring" 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

make rnn_training