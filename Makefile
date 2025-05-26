.PHONY: clean

DAY_RANGE ?= 15
RAW_DATA_PATH ?= data/raw/raw_data.rds
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

# RNN Hyperparameters
INPUT_SIZE ?=
HIDDEN_SIZE ?=
SITE_FEATURES_SIZE ?=
RNN_TYPE ?= GRU
NUM_LAYERS ?= 1
DROPOUT_RATE ?= 0.2
CONCAT_FEATURES ?= False

load_data: 
	python src/data/load_data.py \
		--input_path=$(RAW_DATA_PATH) \
    	--output_dir=data/raw/

preprocess_features:
	python src/data/preprocess_features.py \
		--input_path=data/raw/raw_data.parquet \
    	--output_dir=data/interim/

pivot_data:
	python src/data/pivot_data.py \
		--input_path=data/interim/clean_feats_data.parquet \
		--output_dir=data/processed/$(THRESHOLD_PCT) \
		--day_range=$(DAY_RANGE) \
		--threshold=$(THRESHOLD) \

data_split:
	python src/data/data_split.py \
        --input_path=data/processed/$(THRESHOLD_PCT)/processed_data.parquet \
    	--output_dir=data/processed/$(THRESHOLD_PCT) \

data_split_RNN:
	python src/data/data_split.py \
        --input_path=data/interim/clean_feats_data.parquet \
    	--output_dir=data/interim \

time_series_train_data:
	python -m src.data.get_time_series \
		--input_path=data/interim/train_data.parquet \
		--output_seq_dir=data/processed/sequences \
		--output_lookup_path=data/processed/train_lookup.parquet \

time_series_test_data:
	python -m src.data.get_time_series \
		--input_path=data/interim/test_data.parquet \
		--output_seq_dir=data/processed/sequences \
		--output_lookup_path=data/processed/test_lookup.parquet \
		--no-compute-norm-stats


logistic_regression_pipeline:
	python src/models/logistic_regression_pipeline.py \
		--feat_select='$(FEAT_SELECT)' \
		--drop_features=$(DROP_FEATURES) \
		--step_rfe=$(STEP_RFE) \
		--num_feats_rfe=$(NUM_FEATS_RFE) \
		--min_num_feats_rfecv=$(MIN_NUM_FEATS_RFECV) \
		--num_folds_rfecv=$(NUM_FOLDS_RFECV) \
		--scoring_rfecv="$(SCORING)" \
		--kwargs_json='{}' \
		--output_dir=models/

random_forest_pipeline:
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

gradient_boosting_pipeline:
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

rnn_pipeline:
	python src/models/rnn.py \
		--input_size=$(INPUT_SIZE) \
		--hidden_size=$(HIDDEN_SIZE) \
		--site_features_size=$(SITE_FEATURES_SIZE) \
		--rnn_type=$(RNN_TYPE) \
		--num_layers=$(NUM_LAYERS) \
		--dropout_rate=$(DROPOUT_RATE) \
		--concat_features=$(CONCAT_FEATURES) \
		--output_dir=models/


cv_tuning:
	python src/training/cv_tuning.py \
		--model_path=models/gbm_model.joblib \
		--training_data=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--test_data=data/processed/$(THRESHOLD_PCT)/test_data.parquet \
		--tuning_method=$(TUNING_METHOD) \

		--param_grid='{"xgbclassifier__n_estimators": [1,10], "xgbclassifier__learning_rate": [0.001,10], "xgbclassifier__max_depth":[1,2]}' \
		--num_iter=$(NUM_ITER) \
		--num_folds=$(NUM_FOLDS) \
		--scoring=$(SCORING) \
		--random_state=$(RANDOM_STATE) \
		--return_results=$(RETURN_RESULTS) \
		--output_dir=models/

test:
	pytest

clean:
	rm -rf data/raw
	rm -rf data/interim
	rm -rf data/processed
	rm -rf models