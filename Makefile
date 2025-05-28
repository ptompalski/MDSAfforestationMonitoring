.PHONY: clean load_data preprocess_features pivot_data data_split_RNN time_series_train_data time_series_test_data \
logistic_regression_pipeline random_forest_pipeline gru_pipeline_site_feats all_models tune_gbm tune_lr tune_rf \
tune_classical_models clean_models clean_data

DAY_RANGE ?= 15
RAW_DATA_PATH ?= data/raw/AfforestationAssessmentDataUBCCapstone.rds
THRESHOLD ?= 0.7
THRESHOLD_PCT = $(shell echo | awk '{printf "%.0f", $(THRESHOLD)*100}')

FEAT_SELECT ?= None
DROP_FEATURES ?= 
STEP_RFE ?= 1
NUM_FEATS_RFE ?= 5
MIN_NUM_FEATS_RFECV ?= 2
NUM_FOLDS_RFECV ?= 5

TUNING_METHOD ?= random
NUM_ITER ?= 2
NUM_FOLDS ?= 2
SCORING ?= f1
RANDOM_STATE ?= 591
RETURN_RESULTS ?= True
PARAM_GRID ?=default

# RNN Hyperparameters
INPUT_SIZE ?= 12 
HIDDEN_SIZE ?= 16
SITE_FEATURES_SIZE ?= 4
RNN_TYPE ?= GRU
NUM_LAYERS ?= 1
DROPOUT_RATE ?= 0.2
CONCAT_FEATURES ?= False

### Targets and Dependencies ###

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

# data_split_RNN
data/interim/train_data.parquet \
data/interim/test_data.parquet: data/interim/clean_feats_data.parquet
	python src/data/data_split.py \
        --input_path=data/interim/clean_feats_data.parquet \
    	--output_dir=data/interim \

# time_series_train_data
data/processed/train_lookup.parquet data/interim/norm_stats.json: data/interim/train_data.parquet
	python -m src.data.get_time_series \
		--input_path=data/interim/train_data.parquet \
		--output_seq_dir=data/processed/sequences \
		--output_lookup_path=data/processed/train_lookup.parquet \
		--norm_stats_path=data/interim/norm_stats.json \
		--compute-norm-stats

# time_series_test_data
data/processed/test_lookup.parquet: data/interim/test_data.parquet data/interim/norm_stats.json
	python -m src.data.get_time_series \
		--input_path=data/interim/test_data.parquet \
		--output_seq_dir=data/processed/sequences \
		--output_lookup_path=data/processed/test_lookup.parquet \
		--no-compute-norm-stats

models/logistic_regression.joblib:
	python src/models/logistic_regression.py \
		--feat_select='$(FEAT_SELECT)' \
		--drop_features=$(DROP_FEATURES) \
		--step_rfe=$(STEP_RFE) \
		--num_feats_rfe=$(NUM_FEATS_RFE) \
		--min_num_feats_rfecv=$(MIN_NUM_FEATS_RFECV) \
		--num_folds_rfecv=$(NUM_FOLDS_RFECV) \
		--scoring_rfecv="$(SCORING)" \
		--kwargs_json='{}' \
		--output_dir=models/

models/random_forest.joblib:
	python src/models/random_forest.py \
		--feat_select='$(FEAT_SELECT)' \
		--drop_features=$(DROP_FEATURES) \
		--step_rfe=$(STEP_RFE) \
		--num_feats_rfe=$(NUM_FEATS_RFE) \
		--min_num_feats_rfecv=$(MIN_NUM_FEATS_RFECV) \
		--num_folds_rfecv=$(NUM_FOLDS_RFECV) \
		--scoring_rfecv="$(SCORING)" \
		--kwargs_json='{}' \
		--output_dir=models/

models/gradient_boosting.joblib:
	python src/models/gradient_boosting.py \
		--feat_select='$(FEAT_SELECT)' \
		--drop_features=$(DROP_FEATURES) \
		--step_rfe=$(STEP_RFE) \
		--num_feats_rfe=$(NUM_FEATS_RFE) \
		--min_num_feats_rfecv=$(MIN_NUM_FEATS_RFECV) \
		--num_folds_rfecv=$(NUM_FOLDS_RFECV) \
		--scoring_rfecv="$(SCORING)" \
		--kwargs_json='{}' \
		--output_dir=models/


#gru_pipeline_site_feats
models/gru_site_feats.pth:
	python src/models/rnn.py \
		--input_size=$(INPUT_SIZE) \
		--hidden_size=$(HIDDEN_SIZE) \
		--site_features_size=$(SITE_FEATURES_SIZE) \
		--rnn_type=$(RNN_TYPE) \
		--num_layers=$(NUM_LAYERS) \
		--dropout_rate=$(DROPOUT_RATE) \
		--concat_features=True \
		--output_dir=models/

#gru_pipeline_site_feats
models/gru_no_site_feats.pth:
	python src/models/rnn.py \
		--input_size=$(INPUT_SIZE) \
		--hidden_size=$(HIDDEN_SIZE) \
		--site_features_size=$(SITE_FEATURES_SIZE) \
		--rnn_type=$(RNN_TYPE) \
		--num_layers=$(NUM_LAYERS) \
		--dropout_rate=$(DROPOUT_RATE) \
		--concat_features=False \
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

### Phony targets ###

# Data loading, interim processing
load_data: data/raw/raw_data.parquet
preprocess_features: data/interim/clean_feats_data.parquet

# Processing and splitting for LR, GBM, RF
pivot_data: data/processed/$(THRESHOLD_PCT)/processed_data.parquet
data_split: data/processed/$(THRESHOLD_PCT)/train_data.parquet data/processed/$(THRESHOLD_PCT)/test_data.parquet

# Processing and splitting for RNN models
data_split_RNN: data/interim/train_data.parquet data/interim/test_data.parquet
time_series_train_data: data/processed/train_lookup.parquet
time_series_test_data: data/processed/test_lookup.parquet

# initialize model pipelines
logistic_regression_pipeline: models/logistic_regression.joblib
random_forest_pipeline: models/random_forest.joblib
gradient_boosting_pipeline: models/gradient_boosting.joblib
gru_pipeline_site_feats: models/gru_site_feats.pth
gru_pipeline_no_site_feats: models/gru_no_site_feats.pth

# construct all models at once
all_models: logistic_regression_pipeline random_forest_pipeline gradient_boosting_pipeline \
gru_pipeline_site_feats gru_pipeline_no_site_feats

# process data for classical model training (LR, GBM, RF)
data_for_classical_models: data_split

# process data for RNN models
data_for_RNN_models: time_series_train_data time_series_test_data

# tune/train classical models
tune_gbm: models/$(THRESHOLD_PCT)/tuned_gradient_boosting.joblib \
models/$(THRESHOLD_PCT)/logs/tuned_gradient_boosting_log.csv

tune_rf: models/$(THRESHOLD_PCT)/tuned_random_forest.joblib \
models/$(THRESHOLD_PCT)/logs/tuned_random_forest_log.csv

tune_lr: models/$(THRESHOLD_PCT)/tuned_logistic_regression.joblib \
models/$(THRESHOLD_PCT)/logs/tuned_logistic_regression_log.csv

tune_classical_models: tune_gbm tune_lr tune_rf

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

clean_models:
	rm -rf models

clean_all: clean_data clean_models