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
        --input_path=data/interim/processed_data50.parquet \
    	--output_dir=data/processed/ 

test:
	pytest

test:
	pytest

clean:
