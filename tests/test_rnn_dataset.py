import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.training.rnn_dataset import AfforestationDataset, collate_fn, dataloader_wrapper
from typing import Tuple


# Test data
SITE_COLS = ['Density', 'Type_Conifer', 'Type_Decidous', 'Age']
SEQ_COLS = ['NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 
            'NBR', 'TCB', 'TCG', 'TCW', 'log_dt', 'neg_cos_DOY']
EXP_SEQ_SHAPE = [(2, 12)]*3 + [(4, 12)]*2 + [(6, 12)] + [(1, 12)]

mock_lookup_df = pd.DataFrame({
    'ID': np.arange(1, 8),
    'PixelID': np.arange(11, 18),
    'SrvvR_Date': np.arange(2001, 2008),
    'Age': [1, 1, 1, 2, 2, 3, 4],
    'Density': [1431.4]*7,
    'Type_Conifer': [0]*7,
    'Type_Decidous': [0]*7,
    'target': np.arange(71, 78),
    'filename': ['mock_seq_1.parquet']*3 + ['mock_seq_2.parquet']*2 + ['mock_seq_3.parquet'] + ['na.parquet'],
})

mock_seq_data_1 = pd.DataFrame({
    'ID': [1]*2,
    "PixelID": [11]*2,
    'ImgDate': ['2001-11-01']*2,
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

mock_batch = [
    (torch.tensor([1, 2, 3, 4]),
     torch.tensor([[1]*12, [2]*12]),
     torch.tensor(71.0)),
    (torch.tensor([5, 6, 7, 8]),
     torch.tensor([[3]*12, [4]*12, [5]*12]),
     torch.tensor(72.0))
]

exp_batch_out = {
    'site_features': torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]]),
    'sequence': torch.tensor([[[3]*12, [4]*12, [5]*12], [[1]*12, [2]*12, [0]*12]]),
    'target': torch.tensor([72.0, 71.0]),
    'sequence_length': torch.tensor([3, 2])
}

@pytest.fixture
def setup_mock_data():
    """
    Create test parquet file
    """
    MOCK_SEQ_DIR = Path('tmp')
    MOCK_SEQ_DIR.mkdir(exist_ok=True)
    MOCK_LOOKUP_PATH = os.path.join(MOCK_SEQ_DIR, 'mock_lookup.parquet')
    dfs = [(mock_lookup_df, 'mock_lookup.parquet'),
           (mock_seq_data_1, 'mock_seq_1.parquet'), 
           (mock_seq_data_2, 'mock_seq_2.parquet'), 
           (mock_seq_data_3, 'mock_seq_3.parquet')]
    for df, name in dfs:
        df.to_parquet(os.path.join(MOCK_SEQ_DIR, name))
    return MOCK_LOOKUP_PATH, MOCK_SEQ_DIR

@pytest.fixture
def mock_dataset(setup_mock_data):
    """
    Create mock dataset.
    """
    MOCK_LOOKUP_PATH, MOCK_SEQ_DIR = setup_mock_data
    return AfforestationDataset(MOCK_LOOKUP_PATH, MOCK_SEQ_DIR, SITE_COLS, SEQ_COLS)



# TESTS: Dataset Class
def test_dataset_init(mock_dataset):
    """
    Test if dataset class initiates properly
    """
    dataset = mock_dataset
    assert isinstance(dataset.original_lookup, pd.DataFrame)
    assert isinstance(dataset.lookup, pd.DataFrame)
    assert dataset.site_cols == SITE_COLS
    assert dataset.seq_cols == SEQ_COLS

def test_dataset_reshuffle(mock_dataset):
    """
    Test if the reshuffle function regenerates a randomised lookup table when called.
    """
    dataset = mock_dataset
    lookup1 = dataset.lookup  # save original shuffled lookup table
    dataset.reshuffle() # reshuffle lookup table
    lookup2 = dataset.lookup
    
    # Allow 3 reshuffling attempts due to limited randomness in mock data.
    fail_count = 0
    while fail_count < 3:     
        if lookup1.equals(lookup2):
            dataset.reshuffle()
            lookup2 = dataset.lookup
            fail_count += 1
        else:
            break
    assert not lookup1.equals(lookup2)
    assert lookup1.shape == lookup2.shape
    assert set(lookup1['ID']) == set(lookup2['ID'])
    assert all(lookup2['Age'] == [1, 1, 1, 2, 2, 3, 4])

def test_dataset_len(mock_dataset):
    """
    Test if function returns the length of lookup table when `len()` is used.
    """
    dataset = mock_dataset
    assert len(dataset) == len(mock_lookup_df)

def test_dataset_getitem(mock_dataset):
    """
    Test if dataset returns data and returns the data in the correct format.
    """
    data = mock_dataset
    assert all(isinstance(idx, Tuple) for idx in data)
    assert all(len(idx) == 3 for idx in data)
    assert all(torch.is_tensor(i) for idx in data for i in idx)
    assert all(len(idx[0]) == len(SITE_COLS) for idx in data)
    assert [idx[1].shape for idx in data] == EXP_SEQ_SHAPE 
    assert data[6][1].equal(torch.zeros((1, len(SEQ_COLS)), dtype=torch.float32)) # Fall back to zero if FileNotFound
    assert all(idx[2].ndim == 0 for idx in data)  # Target tensor is a scalar
    assert [idx[2].item() for idx in data] == data.lookup['target'].to_list()
    


# TESTS : Collation Function
def test_collate_fn():
    """
    Test if collate function process the sequences correctly.
    """
    c_batch = collate_fn(mock_batch)
    assert isinstance(c_batch, dict)
    assert list(c_batch.keys()) == ['site_features', 'sequence', 'target', 'sequence_length']
    assert all(isinstance(i, torch.Tensor) for i in list(c_batch.values()))
    assert all(torch.equal(c_batch[i], exp_batch_out[i]) for i in c_batch)



# TESTS: DataLoader
def test_loader_exceptions(setup_mock_data):
    """
    Test if value error is raised if invalid dtypes are supplied.
    """
    MOCK_LOOKUP_PATH, MOCK_SEQ_DIR = setup_mock_data
    invalid_lookup = ['invalid.parquet']
    invalid_seq = ['tests']
    with pytest.raises(ValueError):
        dataloader_wrapper(invalid_lookup, MOCK_SEQ_DIR, SITE_COLS, SEQ_COLS)  # string or pathlike
        
        dataloader_wrapper(MOCK_LOOKUP_PATH, invalid_seq,
                           SITE_COLS, SEQ_COLS)  # string or pathlike
        
        dataloader_wrapper(MOCK_LOOKUP_PATH, MOCK_SEQ_DIR, site_cols='Age') # list
        dataloader_wrapper(MOCK_LOOKUP_PATH, MOCK_SEQ_DIR, site_cols=['Age', 1])  # list of strings
        
        dataloader_wrapper(MOCK_LOOKUP_PATH, MOCK_SEQ_DIR, seq_cols='NDVI')  # list
        dataloader_wrapper(MOCK_LOOKUP_PATH, MOCK_SEQ_DIR, seq_cols=['NDVI', 1])  # list of strings
        
        dataloader_wrapper(MOCK_LOOKUP_PATH, MOCK_SEQ_DIR,
                           SITE_COLS, SEQ_COLS, batch_size='1')  # integer
        
        dataloader_wrapper(MOCK_LOOKUP_PATH, MOCK_SEQ_DIR,
                           SITE_COLS, SEQ_COLS, num_workers='1')  # integer
        
        dataloader_wrapper(MOCK_LOOKUP_PATH, MOCK_SEQ_DIR,
                           SITE_COLS, SEQ_COLS, pin_memory='True')  # boolean

def test_loader_fallback(setup_mock_data):
    """
    Test if function overwrites invalid integer inputs for batch_size and num_workers.
    """
    MOCK_LOOKUP_PATH, MOCK_SEQ_DIR = setup_mock_data
    dataset, loader = dataloader_wrapper(MOCK_LOOKUP_PATH, MOCK_SEQ_DIR, SITE_COLS, SEQ_COLS, batch_size=-1, num_workers=-1)
    assert loader.batch_size == 32  # Default to 32 if batch_size is not a positive integer.
    assert loader.num_workers == 0  # Default to 0 if num_workers is a negative integer.
    
    shutil.rmtree(MOCK_SEQ_DIR) # Remove temporary directory for test data
