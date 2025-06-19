.PHONY: clean clean_models clean_data clean_results\
load_data preprocess_features pivot_data \
data_split_RNN time_series_train_data time_series_test_data \
logistic_regression_pipeline random_forest_pipeline gru_pipeline_site_feats all_models \
tune_gbm tune_lr tune_rf tune_classical_models \
gradient_boosting_rfecv logistic_regression_rfecv random_forest_rfecv RFECV \
gradient_boosting_shap random_forest_shap logistic_regression_shap SHAP \
errors_gbm errors_lr errors_rf classical_model_evaluation \
gradient_boosting_permute random_forest_permute logistic_regression_permute permutation_importance \
rnn_model rnn_training rnn_evaluation rnn_pipeline \ 


# Variables
DAY_RANGE ?= 15
RAW_DATA_PATH ?= data/raw/AfforestationAssessmentDataUBCCapstone.rds
THRESHOLD ?= 0.7
THRESHOLD_PCT = $(shell echo | awk '{printf "%.0f", $(THRESHOLD)*100}')

FEAT_SELECT ?= None
DROP_FEATURES ?= 
MIN_NUM_FEATS_RFECV ?= 2
NUM_FOLDS_RFECV ?= 5

TUNING_METHOD ?= random
NUM_ITER ?= 2
NUM_FOLDS ?= 2
SCORING ?= f1
RANDOM_STATE ?= 591
RETURN_RESULTS ?= True
PARAM_GRID ?=default


### Targets and Dependencies for Data Preprocessing ###

# load_data
data/raw/raw_data.parquet: $(RAW_DATA_PATH)
	python src/data/load_data.py \
		--input_path=$(RAW_DATA_PATH) \
    	--output_dir=data/raw/

# preprocess_features
data/interim/clean_feats_data.parquet: data/raw/raw_data.parquet
	python src/data/preprocess_features.py \
		--input_path=data/raw/raw_data.parquet \
    	--output_dir=data/interim/

# pivot_data
data/processed/$(THRESHOLD_PCT)/processed_data.parquet: data/interim/clean_feats_data.parquet
	python src/data/pivot_data.py \
		--input_path=data/interim/clean_feats_data.parquet \
		--output_dir=data/processed/$(THRESHOLD_PCT) \
		--day_range=$(DAY_RANGE) \
		--threshold=$(THRESHOLD) \

# data_split
data/processed/$(THRESHOLD_PCT)/train_data.parquet \
data/processed/$(THRESHOLD_PCT)/test_data.parquet: data/processed/$(THRESHOLD_PCT)/processed_data.parquet
	python src/data/data_split.py \
        --input_path=data/processed/$(THRESHOLD_PCT)/processed_data.parquet \
    	--output_dir=data/processed/$(THRESHOLD_PCT) \

# Classical Model Pipelines
models/logistic_regression.joblib:
	python src/models/logistic_regression.py \
		--feat_select='$(FEAT_SELECT)' \
		--drop_features=$(DROP_FEATURES) \
		--kwargs_json='{}' \
		--output_dir=models/

models/random_forest.joblib:
	python src/models/random_forest.py \
		--feat_select='$(FEAT_SELECT)' \
		--drop_features=$(DROP_FEATURES) \
		--kwargs_json='{}' \
		--output_dir=models/

models/gradient_boosting.joblib:
	python src/models/gradient_boosting.py \
		--feat_select='$(FEAT_SELECT)' \
		--drop_features=$(DROP_FEATURES) \
		--kwargs_json='{}' \
		--output_dir=models/

		
# tune_gbm
models/$(THRESHOLD_PCT)/tuned_gradient_boosting.joblib \
models/$(THRESHOLD_PCT)/logs/tuned_gradient_boosting_log.csv: models/gradient_boosting.joblib \
data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/training/cv_tuning.py \
		--model_path=models/gradient_boosting.joblib \
		--training_data=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--tuning_method=$(TUNING_METHOD) \
		--param_grid='$(PARAM_GRID)' \
		--num_iter=$(NUM_ITER) \
		--num_folds=$(NUM_FOLDS) \
		--scoring=$(SCORING) \
		--random_state=$(RANDOM_STATE) \
		--return_results=$(RETURN_RESULTS) \
		--output_dir=models/

# tune_rf
models/$(THRESHOLD_PCT)/tuned_random_forest.joblib \
models/$(THRESHOLD_PCT)/logs/tuned_random_forest_log.csv: models/random_forest.joblib \
data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/training/cv_tuning.py \
		--model_path=models/random_forest.joblib \
		--training_data=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--tuning_method=$(TUNING_METHOD) \
		--param_grid='$(PARAM_GRID)' \
		--num_iter=$(NUM_ITER) \
		--num_folds=$(NUM_FOLDS) \
		--scoring=$(SCORING) \
		--random_state=$(RANDOM_STATE) \
		--return_results=$(RETURN_RESULTS) \
		--output_dir=models/

# tune_lr
models/$(THRESHOLD_PCT)/tuned_logistic_regression.joblib \
models/$(THRESHOLD_PCT)/logs/tuned_logistic_regression_log.csv: models/logistic_regression.joblib \
data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/training/cv_tuning.py \
		--model_path=models/logistic_regression.joblib \
		--training_data=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--tuning_method=$(TUNING_METHOD) \
		--param_grid='$(PARAM_GRID)' \
		--num_iter=$(NUM_ITER) \
		--num_folds=$(NUM_FOLDS) \
		--scoring=$(SCORING) \
		--random_state=$(RANDOM_STATE) \
		--return_results=$(RETURN_RESULTS) \
		--output_dir=models/

# evaluate_gbm
results/$(THRESHOLD_PCT)/.gradient_boosting_evaluation.stamp: models/$(THRESHOLD_PCT)/tuned_gradient_boosting.joblib \
data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/evaluation/error_metrics.py \
		--tuned_model_path=models/$(THRESHOLD_PCT)/tuned_gradient_boosting.joblib \
		--training_data_path=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=results/$(THRESHOLD_PCT)
	touch $@

# evaluate_lr 
results/$(THRESHOLD_PCT)/.logistic_regression_evaluation.stamp: models/$(THRESHOLD_PCT)/tuned_logistic_regression.joblib \
data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/evaluation/error_metrics.py \
		--tuned_model_path=models/$(THRESHOLD_PCT)/tuned_logistic_regression.joblib \
		--training_data_path=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=results/$(THRESHOLD_PCT)
	touch $@

# evaluate_rf
results/$(THRESHOLD_PCT)/.random_forest_evaluation.stamp: models/$(THRESHOLD_PCT)/tuned_random_forest.joblib \
data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/evaluation/error_metrics.py \
		--tuned_model_path=models/$(THRESHOLD_PCT)/tuned_random_forest.joblib \
		--training_data_path=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=results/$(THRESHOLD_PCT)
	touch $@

## Feature Selection ##

# RFECV model pipeline constructors
# gbm_rfecv_pipeline
models/gradient_boosting_rfecv.joblib:
	python src/models/gradient_boosting.py \
		--feat_select=RFECV \
		--drop_features=$(DROP_FEATURES) \
		--min_num_feats_rfecv=$(MIN_NUM_FEATS_RFECV) \
		--num_folds_rfecv=$(NUM_FOLDS_RFECV) \
		--scoring_rfecv="$(SCORING)" \
		--kwargs_json='{}' \
		--output_dir=models/

# rf_rfecv_pipeline
models/random_forest_rfecv.joblib:
	python src/models/random_forest.py \
		--feat_select=RFECV \
		--drop_features=$(DROP_FEATURES) \
		--min_num_feats_rfecv=$(MIN_NUM_FEATS_RFECV) \
		--num_folds_rfecv=$(NUM_FOLDS_RFECV) \
		--scoring_rfecv="$(SCORING)" \
		--kwargs_json='{}' \
		--output_dir=models/

# lr_rfecv_pipeline
models/logistic_regression_rfecv.joblib:
	python src/models/logistic_regression.py \
		--feat_select=RFECV \
		--drop_features=$(DROP_FEATURES) \
		--min_num_feats_rfecv=$(MIN_NUM_FEATS_RFECV) \
		--num_folds_rfecv=$(NUM_FOLDS_RFECV) \
		--scoring_rfecv="$(SCORING)" \
		--kwargs_json='{}' \
		--output_dir=models/

# Train RFECV models

# GBM
models/$(THRESHOLD_PCT)/fitted_gradient_boosting_rfecv.joblib: models/gradient_boosting_rfecv.joblib \
data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/training/RFE_trainer.py \
		--model_path=models/gradient_boosting_rfecv.joblib \
		--training_data=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=models/$(THRESHOLD_PCT)

# RF
models/$(THRESHOLD_PCT)/fitted_random_forest_rfecv.joblib: models/random_forest_rfecv.joblib \
data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/training/RFE_trainer.py \
		--model_path=models/random_forest_rfecv.joblib \
		--training_data=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=models/$(THRESHOLD_PCT)

# LR
models/$(THRESHOLD_PCT)/fitted_logistic_regression_rfecv.joblib: models/logistic_regression_rfecv.joblib \
data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/training/RFE_trainer.py \
		--model_path=models/logistic_regression_rfecv.joblib  \
		--training_data=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=models/$(THRESHOLD_PCT)

# SHAP importance

# gradient_boosting_shap
models/$(THRESHOLD_PCT)/fitted_gradient_boosting_shap.joblib: data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/models/feat_selection.py \
		--estimator=gbm \
		--method=SHAP \
		--drop_features=$(DROP_FEATURES) \
		--input_path=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=models/$(THRESHOLD_PCT) \

# random_forest_shap
models/$(THRESHOLD_PCT)/fitted_random_forest_shap.joblib: data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/models/feat_selection.py \
		--estimator=rf \
		--method=SHAP \
		--drop_features=$(DROP_FEATURES) \
		--input_path=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=models/$(THRESHOLD_PCT) \

# logistic_regression_shap
models/$(THRESHOLD_PCT)/fitted_logistic_regression_shap.joblib: data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/models/feat_selection.py \
		--estimator=lr \
		--method=SHAP \
		--drop_features=$(DROP_FEATURES) \
		--input_path=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=models/$(THRESHOLD_PCT) \

# Permutation Importance

# gradient_boosting_permute
models/$(THRESHOLD_PCT)/fitted_gradient_boosting_permute.joblib: data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/models/feat_selection.py \
		--estimator=gbm \
		--method=permute \
		--drop_features=$(DROP_FEATURES) \
		--input_path=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=models/$(THRESHOLD_PCT) \

# random_forest_permute
models/$(THRESHOLD_PCT)/fitted_random_forest_permute.joblib: data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/models/feat_selection.py \
		--estimator=rf \
		--method=permute \
		--drop_features=$(DROP_FEATURES) \
		--input_path=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=models/$(THRESHOLD_PCT) \


# logistic_regression_permute
models/$(THRESHOLD_PCT)/fitted_logistic_regression_permute.joblib: data/processed/$(THRESHOLD_PCT)/train_data.parquet
	python src/models/feat_selection.py \
		--estimator=lr \
		--method=permute \
		--drop_features=$(DROP_FEATURES) \
		--input_path=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--output_dir=models/$(THRESHOLD_PCT) \

### Phony targets for classical models ###

# Data loading, interim processing
load_data: data/raw/raw_data.parquet
preprocess_features: data/interim/clean_feats_data.parquet

# Processing and splitting for LR, GBM, RF
pivot_data: data/processed/$(THRESHOLD_PCT)/processed_data.parquet
data_split: data/processed/$(THRESHOLD_PCT)/train_data.parquet data/processed/$(THRESHOLD_PCT)/test_data.parquet

# initialize model pipelines
logistic_regression_pipeline: models/logistic_regression.joblib
random_forest_pipeline: models/random_forest.joblib
gradient_boosting_pipeline: models/gradient_boosting.joblib

# tune/train classical models
tune_gbm: models/$(THRESHOLD_PCT)/tuned_gradient_boosting.joblib \
models/$(THRESHOLD_PCT)/logs/tuned_gradient_boosting_log.csv

tune_rf: models/$(THRESHOLD_PCT)/tuned_random_forest.joblib \
models/$(THRESHOLD_PCT)/logs/tuned_random_forest_log.csv

tune_lr: models/$(THRESHOLD_PCT)/tuned_logistic_regression.joblib \
models/$(THRESHOLD_PCT)/logs/tuned_logistic_regression_log.csv

# evaluate performance of tuned classsical models
evaluate_gbm: results/$(THRESHOLD_PCT)/.gradient_boosting_evaluation.stamp
evaluate_rf: results/$(THRESHOLD_PCT)/.random_forest_evaluation.stamp
evaluate_lr: results/$(THRESHOLD_PCT)/.logistic_regression_evaluation.stamp

# run RFECV feature selection
gradient_boosting_rfecv: models/$(THRESHOLD_PCT)/fitted_gradient_boosting_rfecv.joblib
logistic_regression_rfecv: models/$(THRESHOLD_PCT)/fitted_logistic_regression_rfecv.joblib
random_forest_rfecv: models/$(THRESHOLD_PCT)/fitted_random_forest_rfecv.joblib

# run SHAP feature selection
gradient_boosting_shap: models/$(THRESHOLD_PCT)/fitted_gradient_boosting_shap.joblib
random_forest_shap: models/$(THRESHOLD_PCT)/fitted_random_forest_shap.joblib
logistic_regression_shap: models/$(THRESHOLD_PCT)/fitted_logistic_regression_shap.joblib

# run permutation importance feature selection
gradient_boosting_permute: models/$(THRESHOLD_PCT)/fitted_gradient_boosting_permute.joblib
random_forest_permute: models/$(THRESHOLD_PCT)/fitted_random_forest_permute.joblib
logistic_regression_permute: models/$(THRESHOLD_PCT)/fitted_logistic_regression_permute.joblib

# construct all classical models at once
all_classical_models: logistic_regression_pipeline random_forest_pipeline gradient_boosting_pipeline 

# process data for classical model training (LR, GBM, RF)
data_for_classical_models: data_split

# Run RFE on all models
RFE: gradient_boosting_rfecv logistic_regression_rfecv random_forest_rfecv

# run SHAP for all models:
SHAP: gradient_boosting_shap random_forest_shap logistic_regression_shap

# run permutation importance on all models
permutation_importance: gradient_boosting_permute random_forest_permute logistic_regression_permute

# tune all models
tune_classical_models: tune_gbm tune_lr tune_rf

# get error metrics for classical models
classical_model_evaluation: evaluate_gbm evaluate_lr evaluate_rf

# RNN Models

# Data Processing for RNN Models
# data_split_RNN
data/interim/train_data.parquet \
data/interim/test_data.parquet: data/interim/clean_feats_data.parquet
	python src/data/data_split.py \
        --input_path=data/interim/clean_feats_data.parquet \
    	--output_dir=data/interim \

# time_series_train_data
data/processed/train_lookup.parquet \
 data/interim/norm_stats.json: data/interim/train_data.parquet
	python -m src.data.get_time_series \
		--input_path=data/interim/train_data.parquet \
		--output_seq_dir=data/processed/sequences \
		--output_lookup_path=data/processed/train_lookup.parquet \
		--norm_stats_path=data/interim/norm_stats.json \
		--compute-norm-stats

# time_series_test_data
data/processed/test_lookup.parquet data/processed/valid_lookup.parquet: data/interim/test_data.parquet data/interim/norm_stats.json
	python -m src.data.get_time_series \
		--input_path=data/interim/test_data.parquet \
		--output_seq_dir=data/processed/sequences \
		--output_lookup_path=data/processed/test_lookup.parquet \
		--no-compute-norm-stats

# Run data processing and splitting for RNN models
data_split_RNN: data/interim/train_data.parquet data/interim/test_data.parquet
time_series_train_data: data/processed/train_lookup.parquet
time_series_test_data: data/processed/test_lookup.parquet data/processed/valid_lookup.parquet
data_for_RNN_models: time_series_train_data time_series_test_data

# Variables for RNN Model Pipeline
# rnn_odel
INPUT_SIZE ?= 12 
HIDDEN_SIZE ?= 32
LINEAR_SIZE ?= 32
SITE_FEATURES_SIZE ?= 4
RNN_TYPE ?= 
NUM_LAYERS ?= 1
DROPOUT_RATE ?= 0.2
CONCAT_FEATURES ?= False
RNN_PIPELINE_PATH ?= 

# rnn_training 
RNN_PIPELINE_PATH ?=
TRAINED_RNN_OUTPUT_PATH ?=
LR ?= 0.001
BATCH_SIZE ?= 64
EPOCHS ?= 10
PATIENCE ?= 5
NUM_WORKERS ?= 0
SITE_COLS ?= Density,Type_Conifer,Type_Decidous,Age
SEQ_COLS ?=NDVI,SAVI,MSAVI,EVI,EVI2,NDWI,NBR,TCB,TCG,TCW,log_dt,neg_cos_DOY

# rnn_evaluation
TRAINED_RNN_PATH ?= 

# RNN Model Pipeline
# Initialise RNN Model
rnn_model:
	python src/models/rnn.py \
		--input_size=$(INPUT_SIZE) \
		--hidden_size=$(HIDDEN_SIZE) \
		--linear_size=$(LINEAR_SIZE) \
		--site_features_size=$(SITE_FEATURES_SIZE) \
		--rnn_type=$(RNN_TYPE) \
		--num_layers=$(NUM_LAYERS) \
		--dropout_rate=$(DROPOUT_RATE) \
		--concat_features=$(CONCAT_FEATURES) \
		--output_path=$(RNN_PIPELINE_PATH)

# Train RNN Model
rnn_training: $(RNN_PIPELINE_PATH)
	python src/training/rnn_train.py \
		--model_path=$(RNN_PIPELINE_PATH)\
		--output_path=$(TRAINED_RNN_OUTPUT_PATH) \
		--data_dir=data/processed/sequences/ \
		--lookup_dir=data/processed/ \
		--lr=$(LR) \
		--batch_size=$(BATCH_SIZE) \
		--epochs=$(EPOCHS) \
		--patience=$(PATIENCE) \
		--num_workers=$(NUM_WORKERS) \
		--site_cols=$(SITE_COLS) \
		--seq_cols=$(SEQ_COLS)

# Run RNN model pipeline
rnn_pipeline: rnn_model rnn_training

# Run evaluation on trained RNN model
rnn_evaluation: $(TRAINED_RNN_PATH)
	python src/evaluation/rnn_evaluation.py \
		--trained_model_path=$(TRAINED_RNN_PATH) \
		--eval_output_path=$(EVAL_OUTPUT_PATH) \
		--lookup_dir=data/processed/ \
		--seq_dir=data/processed/sequences/ \
		--threshold=$(THRESHOLD) \
		--batch_size=$(BATCH_SIZE) \
		--num_workers=$(NUM_WORKERS)

test:
	pytest

clean_data:
	rm -rf data/raw/raw_data.parquet 
	rm -rf data/interim
	rm -rf data/processed
	mkdir data/interim
	touch data/interim/.gitkeep
	mkdir data/processed
	touch data/processed/.gitkeep

clean_results:
	rm -rf results

clean_models:
	rm -rf models

clean_all: clean_data clean_models clean_results