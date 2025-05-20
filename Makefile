.PHONY: clean

all: run_pipeline

load_data: 
	python src/data/load_data.py \
		--input_path=data/raw/raw_data.rds \
    	--output_dir=data/raw/

clean_data:
	python src/data/preprocess_features.py \
		--input_path=data/raw/raw_data.parquet \
    	--output_dir=data/processed/

preprocess_data:
	python src/data/pivot_data.py \
		--input_path=data/processed/clean_feats_data.parquet \
		--output_dir=data/interim/ \
		--day_range=15 \
		--threshold=0.5

data_split:
	python src/data/data_split.py \
        --input_path=data/interim/processed_data50.0.parquet \
    	--output_dir=data/processed/ 

random_forest_pipeline:
	python src/models/random_forest.py \
		--feat_select='None' \
		--drop_features= \
		--step_rfe=1 \
		--num_feats_rfe=5 \
		--min_num_feats_rfecv=2 \
		--num_folds_rfecv=5 \
		--scoring_rfecv="f1" \
		--kwargs_json='{}' \
		--output_dir=models/

gradient_boosting_pipeline:
	python src/models/gradient_boosting.py \
		--feat_select='None' \
		--drop_features= \
		--step_rfe=1 \
		--num_feats_rfe=5 \
		--min_num_feats_rfecv=2 \
		--num_folds_rfecv=5 \
		--scoring_rfecv="f1" \
		--kwargs_json='{}' \
		--output_dir=models/

cv_tuning:
	python src/training/cv_tuning.py \
		--model_path=models/gbm_model.joblib \
		--training_data=data/processed/train_data.parquet \
		--test_data=data/processed/test_data.parquet \
		--tuning_method=random \
		--param_grid='{"xgbclassifier__n_estimators": [1,10], "xgbclassifier__learning_rate": [0.001,10], "xgbclassifier__max_depth":[1,2]}' \
		--num_iter=2 \
		--num_folds=2 \
		--scoring="f1" \
		--random_state=591 \
		--return_results=True \
		--output_dir=models/

test:
	pytest

clean:
