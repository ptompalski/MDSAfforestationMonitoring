#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=st-alexrod6-1          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10    
#SBATCH --mem=64G                   
#SBATCH -e logs/cv_tune_%j.err       # STDERR log
#SBATCH -o logs/cv_tune_%j.out       # STDOUT log
#SBATCH --mail-user=EMAIL        # <-- Specify email
#SBATCH --job-name=cv_tune      # <-- Specify job name
#SBATCH --mail-type=ALL

#### DEFINE INITIAL PARAMETERS HERE BEFORE RUNNING #####

NUM_ITER=2000 # number of random search iterations 
NUM_FOLDS=5 # number of CV folds

########################################################

# unbuffer print output
export PYTHONUNBUFFERED=1

echo '[1/8] Loading modules...'
module load gcc
module load cuda
module load miniconda3

echo '[2/8] Switching to local scratch directory...'
cd "$TMPDIR" || { echo "Failed to cd into $TMPDIR"; exit 2; }

echo "[3/8] Checking for repo archive at: $REPO_TGZ"

# Set repo source path based on scratch directory The repo should be cloned here and at least contain the raw RDS data.
# You may want to remove the processed sequence data before compressing to tgz, as it can take a while if you don't.
# This will compress the repo to repo.tgz if not already done so.

REPO_SOURCE_DIR="${SCRATCH_RUN_DIR}/MDSAfforestationMonitoring"

# if github repo doesnt exist, throw error
if [[ ! -d "$REPO_SOURCE_DIR" ]]; then
    echo "[3.1/9] ERROR: Source directory $REPO_SOURCE_DIR does not exist"; exit 31
fi

# create archive if it doesn't exist or is old
if [[ ! -f "$REPO_TGZ" || "$(find "$REPO_SOURCE_DIR" -type f -newer "$REPO_TGZ" | wc -l)" -gt 0 ]]; then
    echo "[3.2/9] Creating or updating archive: $REPO_TGZ"
    tar -czf "$REPO_TGZ" -C "$SCRATCH_RUN_DIR" MDSAfforestationMonitoring || { echo "Archiving failed"; exit 32; }
else
    echo "[3.2/9] Archive exists and is up-to-date — skipping compression"
fi

echo "[3.3/9] Extracting archive: $REPO_TGZ"
tar -xzf "$REPO_TGZ" || { echo "Extraction failed"; exit 33; }

cd MDSAfforestationMonitoring || { echo "Failed to cd into extracted repo"; exit 3; }


echo '[4/9] Activating Conda environment...'
eval "$(conda shell.bash hook)"
conda activate mds-afforest-dev


echo '[5/9] Prepare data'
make data_for_classical_models THRESHOLD=0.5
make data_for_classical_models THRESHOLD=0.55
make data_for_classical_models THRESHOLD=0.6
make data_for_classical_models THRESHOLD=0.65
make data_for_classical_models THRESHOLD=0.7
make data_for_classical_models THRESHOLD=0.75
make data_for_classical_models THRESHOLD=0.8

echo '[6/9] build models'
make all_classical_models


echo '[7/9] Run hyperparameter tuning'
make tune_classical_models THRESHOLD=0.5 NUM_ITER=${NUM_ITER} NUM_FOLDS=${NUM_FOLDS}
make tune_classical_models THRESHOLD=0.55 NUM_ITER=${NUM_ITER} NUM_FOLDS=${NUM_FOLDS}
make tune_classical_models THRESHOLD=0.6 NUM_ITER=${NUM_ITER} NUM_FOLDS=${NUM_FOLDS}
make tune_classical_models THRESHOLD=0.65 NUM_ITER=${NUM_ITER} NUM_FOLDS=${NUM_FOLDS}
make tune_classical_models THRESHOLD=0.7 NUM_ITER=${NUM_ITER} NUM_FOLDS=${NUM_FOLDS}
make tune_classical_models THRESHOLD=0.75 NUM_ITER=${NUM_ITER} NUM_FOLDS=${NUM_FOLDS}
make tune_classical_models THRESHOLD=0.8 NUM_ITER=${NUM_ITER} NUM_FOLDS=${NUM_FOLDS}

echo '[8/9] Retrieve evaluation metrics'
make classical_model_evaluation THRESHOLD=0.5
make classical_model_evaluation THRESHOLD=0.55
make classical_model_evaluation THRESHOLD=0.6
make classical_model_evaluation THRESHOLD=0.65
make classical_model_evaluation THRESHOLD=0.7
make classical_model_evaluation THRESHOLD=0.75
make classical_model_evaluation THRESHOLD=0.8

echo '[9/9] Packaging results'
tar -czf "$RESULT_TGZ" ./models ./results || { echo "Failed to package results"; exit 8; }

echo "Job complete. Trained model saved to $RESULT_TGZ"

exit 0;