import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import os
import sys
from src.training.rnn_dataset import AfforestationDataset, collate_fn, dataloader_wrapper
from typing import Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Test data
SITE_COLS = ['Density', 'Type_Conifer', 'Type_Decidous', 'Type_Mixed', 'Age']
SEQ_COLS = ['DOY', 'neg_cos_DOY', 'log_dt', 'NDVI',
            'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 
            'NBR', 'TCB', 'TCG', 'TCW']
EXP_SEQ_SHAPE = [(2, 13)]*3 + [(4, 13)]*2 + [(6, 13)] + [(1, 13)]

mock_lookup_df = pd.DataFrame({
    'ID': np.arange(1, 8),
    'PixelID': np.arange(11, 18),
    'SrvvR_Date': np.arange(2001, 2008),
    'Age': [1, 1, 1, 2, 2, 3, 4],
    'Density': [1431.4]*7,
    'Type_Conifer': [0]*7,
    'Type_Decidous': [0]*7,
    'Type_Mixed': [1]*7,
    'target': np.arange(71, 78),
    'filename': ['mock_seq_1.parquet']*3 + ['mock_seq_2.parquet']*2 + ['mock_seq_3.parquet'] + ['na.parquet'],
})

mock_seq_data_1 = pd.DataFrame({
    'ID': [1]*2,
    "PixelID": [11]*2,
    'ImgDate': ['2001-11-01']*2,
    'DOY': np.arange(1, 3),
    'NDVI':  [0.8, 0.6],
    'SAVI':  [0.8, 0.6],
    'MSAVI': [0.8, 0.6],
    'EVI':   [0.8, 0.6],
    'EVI2':  [0.8, 0.6],
    'NDWI':  [0.8, 0.6],
    'NBR':   [0.8, 0.6],
    'TCB':   [0.8, 0.6],
    'TCG':   [0.8, 0.6],
    'TCW':   [0.8, 0.6],
    'log_dt': np.arange(51, 53),
    'neg_cos_DOY': np.arange(21, 23)
})

mock_seq_data_2 = pd.DataFrame({
    'ID': [1]*4,
    "PixelID": [11]*4,
    'ImgDate': ['2001-11-01']*4,
    'DOY': np.arange(3, 7),
    'NDVI':  [0.8, 0.6]*2,
    'SAVI':  [0.8, 0.6]*2,
    'MSAVI': [0.8, 0.6]*2,
    'EVI':   [0.8, 0.6]*2,
    'EVI2':  [0.8, 0.6]*2,
    'NDWI':  [0.8, 0.6]*2,
    'NBR':   [0.8, 0.6]*2,
    'TCB':   [0.8, 0.6]*2,
    'TCG':   [0.8, 0.6]*2,
    'TCW':   [0.8, 0.6]*2,
    'log_dt': np.arange(53, 57),
    'neg_cos_DOY': np.arange(23, 27)
})

mock_seq_data_3 = pd.DataFrame({
    'ID': [1]*6,
    "PixelID": [11]*6,
    'ImgDate': ['2001-11-01']*6,
    'DOY': np.arange(7, 13),
    'NDVI':  [0.8, 0.6]*3,
    'SAVI':  [0.8, 0.6]*3,
    'MSAVI': [0.8, 0.6]*3,
    'EVI':   [0.8, 0.6]*3,
    'EVI2':  [0.8, 0.6]*3,
    'NDWI':  [0.8, 0.6]*3,
    'NBR':   [0.8, 0.6]*3,
    'TCB':   [0.8, 0.6]*3,
    'TCG':   [0.8, 0.6]*3,
    'TCW':   [0.8, 0.6]*3,
    'log_dt': np.arange(57, 63),
    'neg_cos_DOY': np.arange(27, 33)
})
