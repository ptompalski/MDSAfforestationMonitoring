.PHONY: clean

all: run_pipeline

load_data: 
	python src/data/load_data.py \
		--input_path=data/raw/raw_data.rds \
    	--output_dir=data/raw/

preprocess_feature:
	python src/data/preprocess_features.py \
		--input_path=data/raw/raw_data.parquet \
    	--output_dir=data/interim/

data_split:
	python src/data/data_split.py \
        --input_path=data/interim/clean_feats_data.parquet \
    	--output_dir=data/processed/

clean:
