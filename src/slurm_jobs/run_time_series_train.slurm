#!/bin/bash
#SBATCH --time=11:00:00
#SBATCH --account=st-alexro+                   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10                           # 10 CPUs for multiprocessing           
#SBATCH --mem=16G                   
#SBATCH --job-name=ts_test_data
#SBATCH -e logs/slurm-%j.err                         # STDERR log
#SBATCH -o logs/slurm-%j.out                         # STDOUT log
#SBATCH --mail-user=benjamin.frizzell01l@gmail.com   # <--- change to your email
#SBATCH --mail-type=ALL

# Load environment modules
module load gcc
module load miniconda3

# Activate your conda environment
conda activate mds-afforest-dev

# Define a timestamped scratch run directory
TS=$(date +%s)
SCRATCH_RUN_DIR=/scratch/st-alexrod6-1/$USER/afforest_run_$TS

# Copy your GitHub repo to scratch (excluding .git)
rsync -a --exclude='.git' /arc/home/$USER/MDSAfforestationMonitoring "$SCRATCH_RUN_DIR"

# Move to the scratch working directory
cd "$SCRATCH_RUN_DIR"

# Make sure the logs directory exists for SLURM output files
mkdir -p logs

# Run Makefile command
make time_series_train_data