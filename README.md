# MDS Afforestation Monitoring

## Summary

Monitoring afforestation progress across hundreds of remote and ecologically diverse sites in Canada poses significant challenge, particularly due to the weak spectral signals from newly planted trees with sparse canopies in early growth stages. This project seeks to address two key research questions: (1) Can satellite-derived vegetation indices and site-level data be used to accurately predict tree survival over time in large-scale afforestation programs? and (2) What modeling approaches are most effective for this task? Using data from Canada’s 2 Billion Trees initiative, we train and evaluate a suite of machine learning models—including logistic regression, random forests, gradient boosting, and, if time permits, deep learning architectures such as RNNs and LSTMs. Addressing these questions is critical for monitoring afforestation efforts, which are esssential for climate mitigation, biodiversity conservation, and ecosystem restoration.

## Report

This project's report is developed using Quarto, a reproducible publishing system.

### Prerequisites

Ensure Quarto CLI is installed. For installation instructions, visit [Quarto Get Started](https://quarto.org/docs/get-started/).

### Rendering the Report

To generate the proposal report, execute the following command in your terminal:

```bash
quarto render reports/proposal/report.qmd
```

## Scripts

### Prerequisites

- Ensure that you have conda installed. If you don't have conda installed, you can download and install it from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

- Ensure that you have installed GNU Make by referring to the [GNU Make installation guide](https://www.gnu.org/software/make/).

### Environment Setup

To set up the conda environment, run the following command in your terminal:

for development:

```bash
conda env create -f environment-dev.yml
```

for non-development:

```bash
conda env create -f environment.yml
```

### Activating the Environment

After creating the environment, activate it using:

```bash
conda activate mds-afforest
```

### Running the Scripts

To run the scripts, ensure you are in the project directory and the environment is activated. You can then execute the scripts using make. You can find the available scripts in the `Makefile` located in the root directory of the project. The scripts are organized into different sections, such as data processing, model training, and evaluation.

```bash
make <name_of_script>
```

You can also run multiple scripts at once by specifying multiple targets in the command. For example, to run both the data processing and model training scripts, you can use:

```bash
make <script_1> <script_2>
```

Note: some scripts may require additional arguments. You can check the `Makefile` for details on how to use each script and what arguments they accept.

## Description of Scripts and Variables

### Data Processing Parameters

- `DAY_RANGE ?= 15`
  - Specifies the number of days to consider for creating pivoted features. Defaults to 15.
- `RAW_DATA_PATH ?= data/raw/raw_data.rds`
  - Path to the raw input data file. Defaults to `data/raw/raw_data.rds`.
- `THRESHOLD ?= 0.7`
  - A threshold for determining survival status.
- `THRESHOLD_PCT = $(shell echo | awk '{printf "%.0f", $(THRESHOLD)*100}')`
  - Calculates the threshold as a percentage (e.g., 0.7 becomes 70). This is used for naming directories or files.

### Feature Selection Parameters

Note that these variables are specific to feature selection with the Logistic Regression, Gradient Boosting, and Random Forest algorithms only.

- `FEAT_SELECT ?= None`
  - Specifies the feature selection method to be used. Defaults to `None`, meaning no explicit feature selection method is applied by default.
- `DROP_FEATURES ?=`
  - A list of features to be dropped. Defaults to an empty list.
- `STEP_RFE ?= 1`
  - Step size for Recursive Feature Elimination (RFE). Relevant if `FEAT_SELECT` involves RFE. Defaults to 1.
- `NUM_FEATS_RFE ?= 5`
  - Number of features to select when using RFE. Defaults to 5.
- `MIN_NUM_FEATS_RFECV ?= 2`
  - Minimum number of features to consider for RFE with Cross-Validation (RFECV). Defaults to 2.
- `NUM_FOLDS_RFECV ?= 5`
  - Number of cross-validation folds for RFECV. Defaults to 5.

### Model Training and Tuning Parameters

- `TUNING_METHOD ?= random`
  - Method for hyperparameter tuning (e.g., `random` search, `grid` search). Defaults to `random`.
- `NUM_ITER ?= 2`
  - Number of iterations for random search hyperparameter tuning. Defaults to 2.
- `NUM_FOLDS ?= 2`
  - Number of cross-validation folds for model training/tuning. Defaults to 2.
- `SCORING ?= f1`
  - Scoring metric used for evaluating models during tuning and feature selection. Defaults to `f1` score.
- `RANDOM_STATE ?= 591`
  - Random seed for reproducibility. Defaults to 591.
- `RETURN_RESULTS ?= True`
  - Flag to indicate whether to return detailed tuning results. Defaults to `True`.

### RNN Hyperparameters

These variables are specific to the Recurrent Neural Network (RNN) model.

- `INPUT_SIZE ?=`
  - Size of the input features for the RNN.
- `HIDDEN_SIZE ?=`
  - Number of units in the hidden layers of the RNN.
- `SITE_FEATURES_SIZE ?=`
  - Size of site-specific features, if used and concatenated in the RNN.
- `RNN_TYPE ?= GRU`
  - Type of RNN cell to use (e.g., `GRU`, `LSTM`). Defaults to `GRU`.
- `NUM_LAYERS ?= 1`
  - Number of recurrent layers in the RNN. Defaults to 1.
- `DROPOUT_RATE ?= 0.2`
  - Dropout rate for regularization in the RNN. Defaults to 0.2.
- `CONCAT_FEATURES ?= False`
  - Boolean flag indicating whether to concatenate additional features (e.g., site features) in the RNN. Defaults to `False`.

### Data Processing

- `load_data:`
  - Executes `src/data/load_data.py`.
  - **Description**: Loads the raw data from `$(RAW_DATA_PATH)` and saves it to `data/raw/`.
- `preprocess_features:`
  - Executes `src/data/preprocess_features.py`.
  - **Description**: Takes the raw data (expected as `data/raw/raw_data.parquet`), preprocesses features, and saves the output to `data/interim/`.
- `pivot_data:`
  - Executes `src/data/pivot_data.py`.
  - **Description**: Takes the cleaned features from `data/interim/clean_feats_data.parquet`, pivots the data based on `$(DAY_RANGE)` and `$(THRESHOLD)`, and saves the processed data to `data/processed/$(THRESHOLD_PCT)/`.
- `data_split:`
  - Executes `src/data/data_split.py`.
  - **Description**: Splits the processed data from `data/processed/$(THRESHOLD_PCT)/processed_data.parquet` into training and testing sets, saving them in the same directory.

### Model Pipeline

These targets run scripts to train different types of models. They share common feature selection parameters.

- `logistic_regression_pipeline:`
  - Executes `src/models/logistic_regression_pipeline.py`.
  - **Description**: Trains a logistic regression model. It uses feature selection parameters (`FEAT_SELECT`, `DROP_FEATURES`, etc.) and saves the trained model and any associated artifacts to the `models/` directory.
- `random_forest_pipeline:`
  - Executes `src/models/random_forest.py`.
  - **Description**: Trains a random forest model. It uses feature selection parameters and saves the trained model to `models/`.
- `gradient_boosting_pipeline:`
  - Executes `src/models/gradient_boosting.py`.
  - **Description**: Trains a gradient boosting model. It uses feature selection parameters and saves the trained model to `models/`.
- `rnn_pipeline:`
  - Executes `src/models/rnn.py`.
  - **Description**: Creates an instance of RNN model using the specified RNN hyperparameters (`INPUT_SIZE`, `HIDDEN_SIZE`, etc.) and saves the model to `models/`.
- `rnn_training`:
  - Executes `src/training/rnn_training.py`.
  - **Description**: Trains the RNN model on multiple epochs of train data and evaluates the model on test data. Saves the trained model to `models/`.

### Model Tuning

- `cv_tuning:`
  - Executes `src/training/cv_tuning.py`.
  - **Description**: Performs cross-validated hyperparameter tuning for a specified model (defaulting to `models/gbm_model.joblib`). It uses training and test data from `data/processed/$(THRESHOLD_PCT)/`, applies the specified `TUNING_METHOD`, `PARAM_GRID`, and other tuning parameters, and saves the results to `models/`.
    

### Utility

- `test:`
  - Executes `pytest`.
  - **Description**: Runs automated tests for the project using pytest.
- `clean:`
  - Executes `rm -rf` commands.
  - **Description**: Removes generated data directories (`data/raw`, `data/interim`, `data/processed`) and the `models/` directory to clean the project workspace.
