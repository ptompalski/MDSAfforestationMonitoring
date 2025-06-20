# Quickstart Guide for MDS Afforest Project

This guide provides a quick overview of the MDS Afforest project of how to set up the development environment, run the pipeline. It is designed to help you get started quickly with the project, whether you are a new contributor or just looking to understand the workflow.

## Environment Setup

This section guides you through setting up a dedicated development environment for the MDS Afforest project. Using a virtual environment helps prevent dependency conflicts and ensures reproducibility. The project leverages Conda for dependency management, offering two environment files: one for standard development and another with GPU support.

### Environment Setup Instructions

These commands will install all required dependencies for development.\
**Note:** Ensure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed before proceeding. Follow these steps to set up and activate the development environment for this project:

1.  **Create the Environment**

    Open a terminal in the project root directory and run one of the following commands:

    - For the standard environment:

      ```bash
      conda env create -f environment-dev.yml
      ```

    - For GPU support (e.g., with Sockeye):

      ```bash
      conda env create -f environment-dev-gpu.yml
      ```

**Note:** The GPU environment is more or less the same as the standard one, but it has different versions of some packages, such as `torch` and `python`, to support GPU compatibility. That being said, it also depends on your GPU type and version too. If you are not using GPU, stick with the standard environment.

2.  **Activate the Environment**

    - For the standard environment:

      ```bash
      conda activate mds-afforest-dev
      ```

    - For GPU support:

      ```bash
      conda activate mds-afforest-dev-gpu
      ```

## Running the Scripts

**Note:** If the target files or instances of the data or models already exist, the scripts will **not overwrite** them. Instead, you will see **nothing to be done** on your console.

To run the scripts, ensure you are in the project directory and the environment is activated. You can then execute the scripts using make. You can find the available scripts in the `Makefile` located in the root directory of the project. A Makefile is a special file containing a set of instructions used by the make build automation tool. In data science and software projects, a Makefile is often used to automate tasks such as data preprocessing, model training, testing, and cleaning up files. The scripts are organized into different sections, such as data processing, model training, and evaluation.

- Data processing scripts are used to prepare the data for analysis, cleaning, and transformation.
- Model training scripts are used to train machine learning models on the prepared data.
- Evaluation scripts are used to assess the performance of the trained models.

### Pre-requisites

Before running the scripts, ensure you have:

- Installed the required environment as described above.
- All necessary data files are available in the expected directories. By default the raw data is not available in the repository, but you can download it from: [Google Drive Link](https://drive.google.com/file/d/1GengsSVG29m0wH9EET1oaVhadv48dgGj/view?usp=drive_link)
- Place the data files in the `data/raw` directory of the project.

### Load the Data

The following command will read the raw data file and convert it into a parquet format, which is more efficient for processing and storage.
To load the data into parquet format, you can use the provided script. Run the following command in the terminal from the project root directory:

```bash
make data/raw/raw_data.parquet RAW_DATA_PATH=data/raw/AfforestationAssessmentDataUBCCapstone.rds
```

### Preprocess the Data

The preprocessing step is crucial for preparing the data for analysis and modelling. It involves cleaning by removing outliers, records before plating dates, dropping uncessary feature columns, creating density feature, and fill in the missing values of species type. The output of this step is a cleaned dataset in parquet format. To preprocess the data, you can use the following command:

```bash
make preprocess_features
```

## Data Processing for Classical Models

This section describes how to process the data for classical machine learning models. According to online literatures, we decided to proceed with 3 models: Random Forest, XGBoost, and Logistic Regression, where logistic regression is served as a baseline model.

### Set the Threshold

In the context of this project, for simpliciy, the threshold is used to determine the high and low survival rates of the afforestation data. The pivoting operation will categorize the survival rates into two groups based on this threshold. This is particularly useful for classification tasks in machine learning models. For example if the threshold is set to 0.7, any survival rate above this value will be considered "high" and below it will be considered "low".

You can set the threshold for the pivoting operation by defining the `THRESHOLD` variable. This variable determines the high and low survival rates. You can set it in the command line when running the pivoting script, as shown below.

```bash
THRESHOLD=0.7
```

### Pivot the data

The pivoting operation is essential for transforming the data into a format suitable for classification tasks. Since we have 7 columns of survival rates, we need to collapse these columns into a single column of survival rates and create a new column indicating whether the survival rate is high or low based on the defined threshold. This will allow us to train classification models effectively. The output of this step will be a pivoted dataset in parquet format.
To pivot the data, you can use the following command:

```bash
make pivot_data THRESHOLD=${THRESHOLD}
```

### Split the Data

The next step is to split the pivoted data into training and test sets. This is crucial for evaluating the performance of the machine learning models. The training set will be used to train the models, while the test set will be used to evaluate their performance on unseen data. The output of this step will be two parquet files: one for the training set and one for the test set.
To split the processed data into training and test sets:

```bash
make data_split THRESHOLD=${THRESHOLD}
```

This will execute the `data_split.py` script to generate the train and test datasets in the specified directory.

### Execute all Preprocessing for Classical Models

The following command will execute all the preprocessing steps for the classical models, including cleaning, setting the threshold, pivoting the data, and splitting it into training and test sets. This will prepare the data for training and evaluating the classical machine learning models.

````bash
**To execute all of these commands at once:**

```bash
make data_for_classical_models THRESHOLD=${THRESHOLD}
````

## Data Processing for RNN Models

Due to temporal nature of the data, we will use Recurrent Neural Networks (RNNs) to model the sequential and seasonal dynamics of vegetation indices. This section describes how to process the data for RNN models, including cleaning, splitting, and generating training and testing sequences.

### Split cleaned data

This splits the partially cleaned dataset into training and testing subsets. The reason why we don't need to set a threshold for the RNN model is that it is designed to handle sequences of data and training RNN model on multiple thresholds does take a lot of time. Therefore the threshold will be applied at the end of RNN model regression training to determine the survival rate classification for evaluation. The output of this step will be two parquet files: one for the training set and one for the test set.
To split the cleaned data into training and testing subsets for RNN models, you can use the following command:

```bash
make data_split_RNN
```

### Generate Training Sequence Data

This step generates the training sequences for the RNN models. It involves creating a lookup table and sequences from the training data, which will be used to train the RNN models. The output of this step will be a lookup table and sequences in parquet format.This will execute the `get_time_series.py` script to generate the training lookup table, the sequences and the `norm_stats.json` file used for standard scaling.
To generate the training time series data for the RNN modelling:

```bash
make time_series_train_data
```

### Generate Testing and Validation Sequence Data

This step generates the testing and validation sequences for the RNN models. It involves creating a lookup table and sequences from the test data, which will be used to evaluate the RNN models. The output of this step will be a lookup table and sequences in parquet format. This will execute the `get_time_series.py` script to generate the validation lookup table and sequences.
The `norm_stats.json` file will be used to standardize the features in the validation data
To generate the test and validation time series data for the RNN modelling:

```bash
make time_series_test_data
```

### Execute all Preprocessing for RNN Models

**To execute all of these commands at once:**

```bash
make data_for_RNN_models
```

## Train Classical Models

This section describes how to create and train the classical machine learning models using the processed data. The training process involves using the prepared datasets to fit the models and save them for later use.
.

### Create classical model instances

To create an instance of the untrained models using the provided pipelines, run the following commands:

- **Logistic Regression:**

  ```bash
  make logistic_regression_pipeline
  ```

  This will create an untrained logistic regression model and save it to `models/logistic_regression.joblib`.

- **Random Forest:**

  ```bash
  make random_forest_pipeline
  ```

  This will create an untrained random forest model and save it to `models/` directory.

- **Gradient Boosting:**
  ```bash
  make gradient_boosting_pipeline
  ```
  This will create an untrained gradient boosting model and save it to `models/` directory.
- **All models**:
  ```bash
  make all_classical_models
  ```
  This will create all the untrained model above and save them to the `models/` directory.

### Fine-tune the classical models

Fine tuning models involving training and hyperparameter optimization to improve their performance. The tuning process will use the training data generated in the previous steps and save the tuned models to the `models/` directory. The tuning process will also evaluate the model performance using cross-validation and return the best parameters and scores.
You can customize hyperparameters for tuning by setting the following variables in your command:

- `TUNING_METHOD`: Specify the tuning method (e.g., `grid` or `random`). grid search will try all combinations of the parameters in the grid, while random search will sample a fixed number of parameter settings from the specified grid.
- `PARAM_GRID`: Define the parameter grid as a string (e.g., `"{'C':[0.1,1,10]}"`). This should be a valid Python dictionary string that specifies the hyperparameters for particular model to tune and their respective values.
- `NUM_ITER`: Number of iterations for randomized search. This is only applicable if you are using random search for tuning. It specifies how many different combinations of hyperparameters to try.
- `NUM_FOLDS`: Number of cross-validation folds. This is used to evaluate the model performance during tuning. It determines how many subsets of the data will be used for cross-validation.
- `SCORING`: Scoring metric (e.g., `accuracy`, `f1`). This specifies the metric used to evaluate the model performance during tuning.
- `RANDOM_STATE`: Random seed for reproducibility. This ensures that the results are consistent across different runs.
- `RETURN_RESULTS`: Set to `True` to return full results. This will return the full results of the tuning process, including the best parameters and scores.

### Set up arguments for tuning

```bash
TUNING_METHOD=random
PARAM_GRID={}
NUM_ITER=20
NUM_FOLDS=5
SCORING=f1
RANDOM_STATE=42
RETURN_RESULTS=True
```

You can tune each classical model separately using the following commands:

- **Tune Gradient Boosting:**

  ```bash
  make tune_gbm \
      TUNING_METHOD=${TUNING_METHOD} \
      PARAM_GRID=${PARAM_GRID} \
      NUM_ITER=${NUM_ITER} \
      NUM_FOLDS=${NUM_FOLDS} \
      SCORING=${SCORING} \
      RANDOM_STATE=${RANDOM_STATE} \
      RETURN_RESULTS=${RETURN_RESULTS} \
      THRESHOLD_PCT=${THRESHOLD_PCT}
  ```

- **Tune Random Forest:**

  ```bash
  make tune_rf \
      TUNING_METHOD=${TUNING_METHOD} \
      PARAM_GRID=${PARAM_GRID} \
      NUM_ITER=${NUM_ITER} \
      NUM_FOLDS=${NUM_FOLDS} \
      SCORING=${SCORING} \
      RANDOM_STATE=${RANDOM_STATE} \
      RETURN_RESULTS=${RETURN_RESULTS} \
      THRESHOLD_PCT=${THRESHOLD_PCT}
  ```

- **Tune Logistic Regression:**

  ```bash
  make tune_lr \
      TUNING_METHOD=${TUNING_METHOD} \
      PARAM_GRID=${PARAM_GRID} \
      NUM_ITER=${NUM_ITER} \
      NUM_FOLDS=${NUM_FOLDS} \
      SCORING=${SCORING} \
      RANDOM_STATE=${RANDOM_STATE} \
      RETURN_RESULTS=${RETURN_RESULTS} \
      THRESHOLD_PCT=${THRESHOLD_PCT}
  ```

- **Tune all models:**
  ```bash
  make tune_classical_models \
      TUNING_METHOD=${TUNING_METHOD} \
      PARAM_GRID=${PARAM_GRID} \
      NUM_ITER=${NUM_ITER} \
      NUM_FOLDS=${NUM_FOLDS} \
      SCORING=${SCORING} \
      RANDOM_STATE=${RANDOM_STATE} \
      RETURN_RESULTS=${RETURN_RESULTS} \
      THRESHOLD_PCT=${THRESHOLD_PCT}
  ```

### Feature selection for Classical Models

Feature selection is an important step in the machine learning pipeline, as it helps to reduce the dimensionality of the data and improve model performance. In this project, we use Recursive Feature Elimination (RFE), SHAP, and permutation importance to select the most important features for each classical model. If you want to learn more, you can find the details in the [`src/models/feature_selection.py`](src/models/feat_selection.py) script.

### Evaluate Classical Models

After training and tuning the classical models, it is essential to evaluate their performance on the test set. This step will provide insights into how well the models generalize to unseen data. The evaluation will include metrics such as accuracy, precision, recall, and F1 score. To learn more about the evaluation process, you can refer to the [`src/evaluation/error_metrics.py`](src/evaluation/error_metrics.py) script. You can also look at our notebook in the `notebooks/` directory for a more detailed analysis of the model evaluation.

## Train Deep Learning Model (RNNs)

This section describes how to create and train Recurrent Neural Networks (RNNs) for modeling the sequential and seasonal dynamics of vegetation indices. RNNs are particularly well-suited for time series data, as they can capture temporal dependencies in the data.

### Set up arguments for RNN model

- `INPUT_SIZE`: Number of input features per time step (default: 12). This means the number of features in each time step of the input sequence. For example, if you have 12 features like NDVI, SAVI, etc., set this to 12.
- `HIDDEN_SIZE`: Number of hidden units in the RNN (default: 16). This is the size of the hidden state in the RNN cell. It determines how much information the RNN can store and process at each time step.
- `SITE_FEATURES_SIZE`: Number of site-level features to concatenate (default: 4). This is the number of additional features that are concatenated to the RNN input at each time step. For example, if you have site-level features like Density, Type_Conifer, Type_Decidous, and Age, set this to 4.
- `RNN_TYPE`: Type of RNN cell to use (`LSTM`, `GRU`). This specifies the type of RNN cell to use in the model. You can choose between LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit) cells, which are both popular choices for handling sequential data.
- `NUM_LAYERS`: Number of stacked RNN layers (default: 1). This is the number of layers in the RNN. Stacking multiple layers can help the model learn more complex patterns in the data.
- `DROPOUT_RATE`: Dropout rate between RNN layers (default: 0.2). This is the dropout rate applied between RNN layers to prevent overfitting. A common value is 0.2, meaning 20% of the neurons will be randomly set to zero during training.
- `CONCAT_FEATURES`: Whether to concatenate site features (`True` or `False`). If set to `True`, the site-level features will be concatenated to the RNN input at each time step. If set to `False`, the site-level features will not be included in the RNN input.
- `RNN_PIPELINE_PATH`: Path to save the trained RNN pipeline.

To set up the arguments for the RNN model, you can run the following commands in your terminal:

```bash
    INPUT_SIZE=12
    HIDDEN_SIZE=16
    SITE_FEATURES_SIZE=4
    RNN_TYPE=LSTM
    NUM_LAYERS=2
    DROPOUT_RATE=0.3
    CONCAT_FEATURES=True
    RNN_PIPELINE_PATH=models/rnn_model.pth
```

### Set up arguments for RNN training

- `LR`: Learning rate for the optimizer (default: 0.01). This is the step size at each iteration while moving toward a minimum of the loss function. A common starting point is 0.01, but you may need to adjust it based on your model's performance.
- `BATCH_SIZE`: Batch size for training (default: 64). This is the number of samples processed before the model's internal parameters are updated. A common choice is 64, but you can adjust it based on your available memory and dataset size.
- `EPOCHS`: Number of training epochs (default: 10). This is the number of complete passes through the training dataset. A typical starting point is 10, but you may need to increase it based on your model's convergence.
- `PATIENCE`: Early stopping patience (default: 5). This is the number of epochs with no improvement in validation loss before training stops. A common value is 5, meaning if the validation loss does not improve for 5 consecutive epochs, training will stop to prevent overfitting.
- `NUM_WORKERS`: Number of workers for data loading (default: 0). This specifies how many subprocesses to use for data loading. Setting it to 0 means that the data will be loaded in the main process, which is suitable for small datasets. For larger datasets, you can increase this value to speed up data loading.
- `SITE_COLS`: Comma-separated list of site-level feature columns (default: Density,Type_Conifer,Type_Decidous,Age).
  These are the additional features that will be concatenated to the RNN input at each time step. For example, if you have site-level features like Density, Type_Conifer, Type_Decidous, and Age, set this to `Density,Type_Conifer,Type_Decidous,Age`.
- `SEQ_COLS`: Comma-separated list of sequential feature columns (default: NDVI,SAVI,MSAVI,EVI,EVI2,NDWI,NBR,TCB,TCG,TCW,log_dt,neg_cos_DOY).
  These are the features that will be used as input to the RNN model at each time step. For example, if you have features like NDVI, SAVI, MSAVI, etc., set this to `NDVI,SAVI,MSAVI,EVI,EVI2,NDWI,NBR,TCB,TCG,TCW,log_dt,neg_cos_DOY`.
- `TRAINED_RNN_OUTPUT_PATH`: Path to save the trained RNN model outputs

To set up the arguments for RNN training, you can run the following commands in your terminal:

```bash
    LR=0.01
    BATCH_SIZE=64
    EPOCHS=10
    PATIENCE=5
    NUM_WORKERS=0
    SITE_COLS=Density,Type_Conifer,Type_Decidous,Age
    SEQ_COLS=NDVI,SAVI,MSAVI,EVI,EVI2,NDWI,NBR,TCB,TCG,TCW,log_dt,neg_cos_DOY
    TRAINED_RNN_OUTPUT_PATH=models/trained_rnn_model.pth
```

### Create RNN model instance

The following command will create the RNN model instance with the specified parameters. It will initialize the model architecture and prepare it for training.
To initialize the RNN model, run the following command:

```bash
make rnn_model \
    INPUT_SIZE=${INPUT_SIZE} \
    HIDDEN_SIZE=${HIDDEN_SIZE} \
    SITE_FEATURES_SIZE=${SITE_FEATURES_SIZE} \
    RNN_TYPE=${RNN_TYPE} \
    NUM_LAYERS=${NUM_LAYERS} \
    DROPOUT_RATE=${DROPOUT_RATE} \
    CONCAT_FEATURES=${CONCAT_FEATURES} \
    RNN_PIPELINE_PATH=${RNN_PIPELINE_PATH}
```

### Train RNN model

The following command will train the RNN model with the specified parameters. It will use the training data generated in the previous steps and save the trained model to the specified output path.
To train the RNN model with the specified parameters, run:

```bash
make rnn_training \
    LR=${LR} \
        BATCH_SIZE=${BATCH_SIZE} \
        EPOCHS=${EPOCHS} \
        PATIENCE=${PATIENCE} \
        NUM_WORKERS=${NUM_WORKERS} \
        SITE_COLS=${SITE_COLS} \
        SEQ_COLS=${SEQ_COLS} \
        RNN_PIPELINE_PATH=${RNN_PIPELINE_PATH} \
        TRAINED_RNN_OUTPUT_PATH=${TRAINED_RNN_OUTPUT_PATH}
```

### Evaluate RNN model

After training the RNN model, it is essential to evaluate its performance on the test set. This step will provide insights into how well the model generalizes to unseen data. The evaluation will include metrics such as accuracy, precision, recall, and F1 score. To learn more about the evaluation process, you can refer to the [`src/evaluation/rnn_evaluation.py`](src/evaluation/rnn_evaluation.py) script. You can also look at our notebook in the `notebooks/` directory for a more detailed analysis of the model evaluation.

## Test

This section describes how to test self-contained scripts in the MDS Afforest project. The tests are designed to ensure that the scripts work as expected and produce the correct outputs. The tests are written using the `pytest` framework, which is a popular testing framework for Python. It's important for maintaining the quality and reliability of the codebase, especially as the project evolves.
To run the tests, you can use the following command:

```bash
make test
```

## Clean Up

To clean up generated data and models, you can use the following commands:

### Clean data files

```bash
make clean_data
```

This will remove the raw, interim, and processed data files and recreate the necessary directories with `.gitkeep` files.

### Clean model files

```bash
make clean_models
```

This will remove all files in the `models` directory.

### Clean all generated files

```bash
make clean_all
```
