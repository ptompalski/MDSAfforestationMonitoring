
import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evaluation.rnn_evaluation import rnn_get_metrics, rnn_get_prediction, rnn_get_metrics_by_age
from src.models.rnn import RNNSurvivalPredictor
from src.training.rnn_dataset import dataloader_wrapper
from src.data.pivot_data import target_to_bin


mock_test_lookup = pd.DataFrame({
    'ID': np.arange(1, 6),
    'PixelID': np.arange(11, 16),
    'SrvvR_Date': np.arange(2001, 2006),
    'Age': [1, 2, 3, 4, 7],
    'Density': [1431.4]*5,
    'Type_Conifer': [0]*5,
    'Type_Decidous': [0]*5,
    'target': np.arange(71, 76),
    'filename': ['mock_seq_1.parquet']*3 + ['mock_seq_2.parquet']*2,
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

mock_pred_df = pd.DataFrame({
    'ID': [1]*22,
    'PixelID': [11]*22,
    'SrvvR_Date': [1]*22,
    'Age': [1, 2, 3, 4, 5, 7, 1, 2, 3, 5, 7, 1, 3, 5, 7, 2, 4, 2, 1, 5, 3, 3],
    'Density': [1431.4]*22,
    'Type_Conifer': [0]*22,
    'Type_Decidous': [0]*22,
    'y_true': [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    'filename': ['name']*22,
    'y_pred' : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    'raw_y_true' : [60, 60, 90, 90, 90, 90, 60, 60, 90, 90, 90, 60, 60, 60, 90, 60, 60, 60, 90, 90, 90, 90],
    'raw_y_pred': [60, 60, 60, 60, 90, 90, 90, 90, 60, 60, 60, 60, 90, 90, 90, 60, 60, 60, 90, 90, 90, 90]
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
    MOCK_TEST_LOOKUP = os.path.join(MOCK_SEQ_DIR, 'mock_test_lookup.parquet')
    dfs = [(mock_test_lookup, 'mock_test_lookup.parquet'),
           (mock_seq_data_1, 'mock_seq_1.parquet'),
           (mock_seq_data_2, 'mock_seq_2.parquet')
            ]
    for df, name in dfs:
        df.to_parquet(os.path.join(MOCK_SEQ_DIR, name))
    return MOCK_TEST_LOOKUP, MOCK_SEQ_DIR


@pytest.fixture
def setup_model(setup_mock_data):
    """
    Setup dummy model, device, test dataset and test dataloader.
    """
    MOCK_TEST_LOOKUP, MOCK_SEQ_DIR = setup_mock_data
    test_dataset, test_dataloader = dataloader_wrapper(
        lookup_dir=MOCK_TEST_LOOKUP,
        seq_dir=MOCK_SEQ_DIR,
        site_cols=site_cols,
        seq_cols=seq_cols
    )
    
    model = RNNSurvivalPredictor(input_size=12,
                                 hidden_size=16,
                                 site_features_size=4,
                                 concat_features=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    shutil.rmtree(MOCK_SEQ_DIR)  # Remove temporary directory for test data
    return model, test_dataset, test_dataloader, device


def test_rnn_get_prediction(setup_model):
    '''
    Test to make sure output of get_prediction is as expected
    '''
    model, test_dataset, test_dataloader, device = setup_model
    df = rnn_get_prediction(model, test_dataloader, test_dataset, 0.7, device)
    
    exp_y_true = target_to_bin(test_dataset.lookup, 0.7)
    exp_cols = test_dataset.lookup.columns.to_list()
    exp_cols.remove('target')
    exp_cols += ['y_true', 'y_pred', 'raw_y_true', 'raw_y_pred']
    
    assert isinstance(df, pd.DataFrame)
    assert all(i in exp_cols for i in df.columns.to_list()), f'{df.columns}'
    assert np.isin(df['y_true'], [0, 1]).all()
    assert np.isin(df['y_pred'], [0, 1]).all()
    assert len(df) == len(test_dataset)
    assert df['y_true'].equals(exp_y_true['target'])
    
def test_rnn_get_metrics():
    """
    Test if function calculates metrics accurately and returns them in the expected dtype.
    """
    df = mock_pred_df
    metrics, conf_matrix = rnn_get_metrics(df)
    
    TP = 6
    TN = 7
    FP = 5
    FN = 4

    exp_precision = TP / (TP + FP)
    exp_recall = TP / (TP + FN)
    exp_accuracy = (TP + TN) / (TP + TN + FP + FN)
    exp_f1 = 2 * exp_precision * exp_recall / (exp_precision + exp_recall)
    exp_f2 = 5 * exp_precision * exp_recall / (4 * exp_precision + exp_recall)
    exp_pct_low = 100 * round(10 / 22, 3)
    exp_pct_high = 100 * round(12 / 22, 3)

    # Test if results are returned in the correct format.
    assert isinstance(metrics, pd.Series)
    assert isinstance(conf_matrix, pd.DataFrame)
    assert metrics.index.to_list() == [
        'F1 Score', 'F2 Score', 'Precision', 'Recall', 'Accuracy', '% Low Risk', '% High Risk']
    assert conf_matrix.shape == (2, 2)

    # Test if metrics values are calculated correctly.
    assert round(exp_f1, 3) == metrics['F1 Score']
    assert round(exp_f2, 3) == metrics['F2 Score']
    assert round(exp_precision, 3) == metrics['Precision']
    assert round(exp_recall, 3) == metrics['Recall']
    assert round(exp_accuracy, 3) == metrics['Accuracy']
    assert exp_pct_low == metrics['% Low Risk']
    assert exp_pct_high == metrics['% High Risk']
    
    # Test if confusion matrix values are calculated correctly.
    assert conf_matrix.iloc[0, 0] == 6
    assert conf_matrix.iloc[0, 1] == 4
    assert conf_matrix.iloc[1, 0] == 5
    assert conf_matrix.iloc[1, 1] == 7


def rnn_get_metrics_by_age():
    """
    Test if function computes separate metrics for each age group.
    """
    df = mock_pred_df
    metrics_age, conf_matrix = rnn_get_metrics_by_age(df)
    exp_cols = ['F1 Score', 'F2 Score', 'Precision', 'Recall', 'Accuracy',
                 '% Low Risk', '% High Risk', 'Number of Records']
    
    assert isinstance(metrics_age, pd.DataFrame)
    assert metrics_age.shape == (6, 8)
    assert metrics_age.index.to_list() == [1, 2, 3, 4, 5, 7]
    assert metrics_age.columns.to_list() == exp_cols
    assert metrics_age['Number of Records'] == [4, 4, 5, 2, 4, 3]
    assert all(metrics_age.notna())
    
    assert isinstance(conf_matrix, dict)
    assert conf_matrix.keys().to_list() == [1, 2, 3, 4, 5, 7]
    assert isinstance(conf_matrix.values(), pd.DataFrame)
