#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=st-alexrod6-1-gpu            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3    
#SBATCH --gpus=1
#SBATCH --mem=32G                   
#SBATCH -e logs/run_%j.err       # STDERR log
#SBATCH -o logs/run_%j.out       # STDOUT log
#SBATCH --mail-user=EMAIL        # <-- Specify email
#SBATCH --job-name=JOB_NAME      # <-- Specify job name
#SBATCH --mail-type=ALL

#### DEFINE INITIAL PARAMETERS HERE BEFORE RUNNING #####

RNN_TYPE=GRU                                # <-- Type of model: GRU or LSTM
NUM_LAYERS=1                                # <-- Number of layers in sequential step
HIDDEN_SIZE=16                              # <-- Dimension of hidden state
CONCAT_FEATURES=True                        # <-- True: concatenate site features, False: sequence data only
EPOCHS=10                                   # <-- Number of epochs to train on
BATCH_SIZE=64                               # <-- Batch size: samples per optimization step
LR=0.001                                    # <-- Learning rate for training

INPUT_MODEL_NAME=in_model                                   # <-- name of input model file
OUTPUT_MODEL_NAME=out_model                                 # <-- name of output model file
RNN_PIPELINE_PATH="models/${INPUT_MODEL_NAME}.pth"          # <-- path to input model (to be trained)
TRAINED_RNN_OUTPUT_PATH="models/${OUTPUT_MODEL_NAME}.pth"   # <-- path to output model (after retraining)

########################################################

# define paths
SCRATCH_RUN_DIR=/scratch/st-alexrod6-1/$USER
REPO_TGZ="${SCRATCH_RUN_DIR}/repo.tgz"
RESULT_TGZ="$SCRATCH_RUN_DIR/results/${OUTPUT_MODEL_NAME}.tgz"
mkdir -p "$SCRATCH_RUN_DIR/results"

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
    echo "[3.1/8] ERROR: Source directory $REPO_SOURCE_DIR does not exist"; exit 31
fi

# create archive if it doesn't exist or is old
if [[ ! -f "$REPO_TGZ" || "$(find "$REPO_SOURCE_DIR" -type f -newer "$REPO_TGZ" | wc -l)" -gt 0 ]]; then
    echo "[3.2/8] Creating or updating archive: $REPO_TGZ"
    tar -czf "$REPO_TGZ" -C "$SCRATCH_RUN_DIR" MDSAfforestationMonitoring || { echo "Archiving failed"; exit 32; }
else
    echo "[3.2/8] Archive exists and is up-to-date — skipping compression"
fi

echo "[3.3/8] Extracting archive: $REPO_TGZ"
tar -xzf "$REPO_TGZ" || { echo "Extraction failed"; exit 33; }

cd MDSAfforestationMonitoring || { echo "Failed to cd into extracted repo"; exit 3; }


echo '[4/8] Activating Conda environment...'
eval "$(conda shell.bash hook)"
conda activate mds-afforest-dev-gpu


echo '[5/8] Prepare data'
make data_for_RNN_models

echo '[6/8] Build model'
# retrain if the model exists, build the model if it doesn't
if [[ ! -f "./${RNN_PIPELINE_PATH}" ]]; then
    echo "$RNN_PIPELINE_PATH does not exist, creating model"
    make rnn_model RNN_TYPE=${RNN_TYPE} NUM_LAYERS=${NUM_LAYERS} HIDDEN_SIZE=${HIDDEN_SIZE} CONCAT_FEATURES=${CONCAT_FEATURES} RNN_PIPELINE_PATH=${RNN_PIPELINE_PATH}

else
    echo "$RNN_PIPELINE_PATH already exists"
fi


echo '[7/8] Run training loop'
make rnn_training EPOCHS=${EPOCHS} LR=${LR} BATCH_SIZE=${BATCH_SIZE} RNN_PIPELINE_PATH=${RNN_PIPELINE_PATH} TRAINED_RNN_OUTPUT_PATH=${TRAINED_RNN_OUTPUT_PATH} || { echo "Training failed"; exit 7; }

echo '[8/8] Packaging results'
tar -czf "$RESULT_TGZ" ./models || { echo "Failed to package results"; exit 8; }

echo "Job complete. Trained model saved to $RESULT_TGZ"

exit 0;