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

load_data: 
	python src/data/load_data.py \
		--input_path=$(RAW_DATA_PATH) \
    	--output_dir=data/raw/

preprocess_features:
	python src/data/preprocess_features.py \
		--input_path=data/raw/raw_data.parquet \
    	--output_dir=data/interim/

pivot_data:
	@echo "THRESHOLD: $(THRESHOLD), THRESHOLD_PCT: $(THRESHOLD_PCT)"
	python src/data/pivot_data.py \
		--input_path=data/interim/clean_feats_data.parquet \
		--output_dir=data/processed/$(THRESHOLD_PCT) \
		--day_range=$(DAY_RANGE) \
		--threshold=$(THRESHOLD) \

data_split:
	python src/data/data_split.py \
        --input_path=data/processed/$(THRESHOLD_PCT)/processed_data.parquet \
    	--output_dir=data/processed/$(THRESHOLD_PCT) \

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
		--scoring_rfecv="$(SCORING_RFECV)" \
		--kwargs_json='{}' \
		--output_dir=models/

cv_tuning:
	python src/training/cv_tuning.py \
		--model_path=models/gbm_model.joblib \
		--training_data=data/processed/$(THRESHOLD_PCT)/train_data.parquet \
		--test_data=data/processed/$(THRESHOLD_PCT)/test_data.parquet \
		--tuning_method=$(TUNING_METHOD)
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
