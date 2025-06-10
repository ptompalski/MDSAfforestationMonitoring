import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
import os
import sys
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.training.rnn_dataset import dataloader_wrapper
from src.models.rnn import RNNSurvivalPredictor
from src.training.rnn_train import train

mock_train_lookup = pd.DataFrame({
    'ID': np.arange(1, 6),
    'PixelID': np.arange(11, 16),
    'SrvvR_Date': np.arange(2001, 2006),
    'Age': [1, 1, 1, 2, 2],
    'Density': [1431.4]*5,
    'Type_Conifer': [0]*5,
    'Type_Decidous': [0]*5,
    'target': np.arange(71, 76),
    'filename': ['mock_seq_1.parquet']*3 + ['mock_seq_2.parquet']*2,
})

mock_valid_lookup = pd.DataFrame({
    'ID': np.arange(6, 8),
    'PixelID': np.arange(16, 18),
    'SrvvR_Date': np.arange(2006, 2008),
    'Age': [3, 4],
    'Density': [1431.4]*2,
    'Type_Conifer': [0]*2,
    'Type_Decidous': [0]*2,
    'target': np.arange(76, 78),
    'filename': ['mock_seq_3.parquet'] + ['mock_seq_4.parquet'],
})

mock_seq_data_1 = pd.DataFrame({
    'ID': [1]*2,
    "PixelID": [11]*2,
    'ImgDate': ['2001-11-01']*2,
    'NDVI':  [0.1, 0.2],
    'SAVI':  [0.1, 0.2],
    'MSAVI': [0.1, 0.2],
    'EVI':   [0.1, 0.2],
    'EVI2':  [0.1, 0.2],
    'NDWI':  [0.1, 0.2],
    'NBR':   [0.1, 0.2],
    'TCB':   [0.1, 0.2],
    'TCG':   [0.1, 0.2],
    'TCW':   [0.1, 0.2],
    'log_dt': np.arange(51, 53),
    'neg_cos_DOY': np.arange(21, 23)
})

mock_seq_data_2 = pd.DataFrame({
    'ID': [1]*4,
    "PixelID": [11]*4,
    'ImgDate': ['2001-11-01']*4,
    'NDVI':  [0.1, 0.2, 0.3, 0.4],
    'SAVI':  [0.1, 0.2, 0.3, 0.4],
    'MSAVI': [0.1, 0.2, 0.3, 0.4],
    'EVI':   [0.1, 0.2, 0.3, 0.4],
    'EVI2':  [0.1, 0.2, 0.3, 0.4],
    'NDWI':  [0.1, 0.2, 0.3, 0.4],
    'NBR':   [0.1, 0.2, 0.3, 0.4],
    'TCB':   [0.1, 0.2, 0.3, 0.4],
    'TCG':   [0.1, 0.2, 0.3, 0.4],
    'TCW':   [0.1, 0.2, 0.3, 0.4],
    'log_dt': np.arange(51, 55),
    'neg_cos_DOY': np.arange(21, 25)
})

mock_seq_data_3 = pd.DataFrame({
    'ID': [1]*6,
    "PixelID": [11]*6,
    'ImgDate': ['2001-11-01']*6,
    'NDVI':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'SAVI':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'MSAVI': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'EVI':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'EVI2':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'NDWI':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'NBR':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'TCB':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'TCG':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'TCW':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'log_dt': np.arange(51, 57),
    'neg_cos_DOY': np.arange(21, 27)
})

mock_seq_data_4 = pd.DataFrame({
    'ID': [1]*8,
    "PixelID": [11]*8,
    'ImgDate': ['2001-11-01']*8,
    'NDVI':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'SAVI':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'MSAVI': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'EVI':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'EVI2':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'NDWI':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'NBR':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'TCB':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'TCG':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'TCW':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'log_dt': np.arange(51, 59),
    'neg_cos_DOY': np.arange(21, 29)
})

site_cols = ['Density', 'Type_Conifer', 'Type_Decidous', 'Age']
seq_cols = ['NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR',
                    'TCB', 'TCG', 'TCW', 'log_dt', 'neg_cos_DOY']

@pytest.fixture
def setup_mock_data():
    """
    Create test parquet file
    """
    MOCK_SEQ_DIR = Path('tmp')
    MOCK_SEQ_DIR.mkdir(exist_ok=True)
    MOCK_TRAIN_LOOKUP = os.path.join(MOCK_SEQ_DIR, 'mock_train_lookup.parquet')
    MOCK_VALID_LOOKUP = os.path.join(MOCK_SEQ_DIR, 'mock_valid_lookup.parquet')
    dfs = [(mock_train_lookup, 'mock_train_lookup.parquet'),
           (mock_valid_lookup, 'mock_valid_lookup.parquet'),
           (mock_seq_data_1, 'mock_seq_1.parquet'),
           (mock_seq_data_2, 'mock_seq_2.parquet'),
           (mock_seq_data_3, 'mock_seq_3.parquet'),
           (mock_seq_data_4, 'mock_seq_4.parquet')]
    for df, name in dfs:
        df.to_parquet(os.path.join(MOCK_SEQ_DIR, name))
    return MOCK_TRAIN_LOOKUP, MOCK_VALID_LOOKUP, MOCK_SEQ_DIR

@pytest.fixture
def setup_trainer(setup_mock_data, capsys):
    """
    Setup dummy model, optimizer, loss function, device, dataset and dataloader.
    """
    MOCK_TRAIN_LOOKUP, MOCK_VALID_LOOKUP, MOCK_SEQ_DIR = setup_mock_data
    train_dataset, train_dataloader = dataloader_wrapper(
    lookup_dir=MOCK_TRAIN_LOOKUP,
    seq_dir=MOCK_SEQ_DIR,
    site_cols=site_cols,
    seq_cols=seq_cols
    )
    valid_dataset, valid_dataloader = dataloader_wrapper(
        lookup_dir=MOCK_VALID_LOOKUP,
        seq_dir=MOCK_SEQ_DIR,
        site_cols=site_cols,
        seq_cols=seq_cols
    )
    model = RNNSurvivalPredictor(input_size=12,
                                 hidden_size=16,
                                 linear_size=16,
                                 site_features_size=4,
                                 concat_features=True)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    model_dict = model.state_dict()
    trained_model = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        train_set=train_dataset,
        valid_set=valid_dataset,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        epochs=5
    )
    captured = capsys.readouterr()
    shutil.rmtree(MOCK_SEQ_DIR)  # Remove temporary directory for test data
    return model_dict, trained_model, captured.out

def test_rnn_train_output(setup_trainer):
    """
    Test if the train function returns the model state dict and outputs the losses for each epoch.
    """
    model_dict, trained_model, output = setup_trainer
    assert isinstance(trained_model, dict)
    assert f'Training Model on 5 epochs on' in output
    assert all(f'Epoch {i}: Train Loss =' in output for i in range(1, 6))

    
