import pytest
import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Test data
MOCK_LOOKUP_PATH = "tests/test_lookup.parquet"
MOCK_SEQ_DIR = "tests"

mock_lookup_data = pd.DataFrame({
    'ID': np.arange(1, 7), "PixelID": np.arange(101, 107), 'Type': ['Decidous']*6,

    'target': [pd.NA]*6})

mock_seq_data_1 = pd.DataFrame({
    'ID': np.arange(1, 7), 
    "PixelID": np.arange(101, 107),
    'ImgDate': pd.date_range(start="2001-01-01", end="2007-01-01", freq='YE'), 
    'DOY': [1]*6,
    'NDVI':  [0.8, 0.6, 1.2, -0.5, -1.1, 0.3],   
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],
    'NBR':   [1.0, -0.6, 0.2, -0.8, 0.3, 0.1],
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],
    'TCG':   [0.6, -0.2, 0.4, 0.5, -1.0, 0.2],
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],
})
