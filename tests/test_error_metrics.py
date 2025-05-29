import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.gradient_boosting import build_gbm_pipeline
from sklearn.model_selection import train_test_split
from src.evaluation.error_metrics import (
    get_validation_preds,get_test_errors,get_valid_roc_curve,get_valid_pr_curve,
    get_conf_matrix,get_error_metrics
)

@pytest.fixture()
def dummy_data():
    '''
    Create sample dataframe fitting to models.
    '''
    sample_data = pd.DataFrame({
        'ID': np.arange(1, 16),
        "PixelID": np.arange(101, 116),
        "Density": [10]*5 + [20]*5 + [30]*5,
        'Type': ['Decidous']*5 + ['Mixed']*5 + ['Conifer']*5,
        "Season": np.arange(2001,2016),
        'Age': [1, 1, 3, 5, 2, 3, 6, 7, 5, 5, 7, 1, 3, 4, 5],
        #'DOY':[1]*15,
        'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3, 0.3, 0.6, 0.7, 0.8, 0.6, 0.2, -0.5, -0.1, 0.3],   
        'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1, 0.1, 0.5, 0.9, 0.9, -0.6, 0.2, -0.8, 0.3, 0.7], 
        'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2, -0.9, 0.5, -0.5, 0.7, -0.9, 0.5, -0.5, 0.2, 0.1],   
        'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0, 0.3, -0.8, -0.2, 0.7, -0.9, 0.5, -0.5, 0.2, 0.1],   
        'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5, 0.9, -0.6, 0.2, 0.7, -0.9, 0.5, -0.5, 0.2, 0.1],  
        'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1, 0.1, 0.1, 0.5, -0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
        'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1, -0.2, 0.4, 0.5, 0.3, -0.8, -0.2, 2.5, 0.6, -0.4],  
        'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7, 0.3, -0.4, 0.8, 0.3, -0.8, -0.2, 2.5, 0.6, -0.4],   
        'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2, -0.2, 0.1, 0.1, 0.3, -0.8, -0.2, 2.5, 0.6, -0.4],   
        'TCW':   [0.3, -0.8, -0.2, 2.5, 0.6, -0.4, 0.9, 0.6, 0.1, -0.2, 2.5, 0.6, -0.4, 0.9, 0.6], 
        'target': [0]*8 + [1]*7,
        'SrvvR_Date': pd.date_range(start="2001-01-01", end="2015-01-01", freq="YS")
    })
    return sample_data

@pytest.fixture
def dummy_model():
    '''
    Create a dummy GBM model for testing predictions.
    '''
    return build_gbm_pipeline(n_estimators=5)

def test_get_preds_and_truth(dummy_model, dummy_data):
    '''
    Test to make sure output of get_preds_and_truth is as expected
    '''
    result = get_validation_preds(dummy_model, dummy_data, num_folds=3)
    assert set(result.keys()) == {'y_pred', 'y_prob', 'y_true'}
    assert len(result['y_pred']) == len(dummy_data)
    assert np.all((0 <= result['y_prob']) & (result['y_prob'] <= 1))

def test_get_valid_roc_curve(dummy_model, dummy_data):
    '''
    Test to ensure output ROC curve is as expected.
    '''
    preds = get_validation_preds(dummy_model, dummy_data, num_folds=3)
    df_roc = get_valid_roc_curve(preds['y_prob'], preds['y_true'])
    assert {'False Positive Rate', 'True Positive Rate', 'Thresholds'}.issubset(df_roc.columns)

def test_get_valid_pr_curve(dummy_model, dummy_data):
    '''
    Test to ensure output PR curve is as expected.
    '''
    preds = get_validation_preds(dummy_model, dummy_data, num_folds=3)
    df_pr = get_valid_pr_curve(preds['y_prob'], preds['y_true'])
    assert {'Precision', 'Recall', 'Thresholds'}.issubset(df_pr.columns)
    
def test_get_conf_matrix(dummy_model, dummy_data):
    preds = get_validation_preds(dummy_model, dummy_data, num_folds=3)
    cm = get_conf_matrix(preds['y_pred'], preds['y_true'])
    assert cm.shape == (2, 2)
    assert all(label in cm.columns for label in ['Predicted Low', 'Predicted High'])

def test_conf_matrix_values():
    '''
    Test to make sure confusion matrix contains accurate values
    '''
    y_pred = np.array([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1])
    y_true = np.array([0,0,1,1,1,1,0,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1])

    M = get_conf_matrix(y_pred,y_true).to_numpy()
    # 6 correct predictions of 0: (0,0)
    assert M[0,0] == 6
    # 7 correct predicitons of 1: (1,1)
    assert M[1,1] == 7
    # 5 incorrect predictions of 0: (1,0)
    assert M[1,0] == 5
    # 4 incorrect predicions of 1: (0,1)
    assert M[0,1] == 4
    
def test_get_error_metrics(dummy_model, dummy_data):
    '''
    Test to ensure type and formatting of get_error_metrics output is correct
    '''
    preds = get_validation_preds(dummy_model, dummy_data, num_folds=3)
    scores = get_error_metrics(preds['y_pred'], preds['y_prob'], preds['y_true'])
    expected_keys = {
        'F1 Score', 'F2 Score', 'Precision', 'Recall', 'Accuracy',
        'AUC', 'AP', '% Low Risk', '% High Risk'
    }
    assert expected_keys.issubset(scores.keys())
    assert all(isinstance(v, float) for v in scores.values())
    
def test_get_test_errors(dummy_model, dummy_data):
    '''
    Test to ensure type and formatting of get_error_metrics output is correct
    '''
    dummy_test = pd.DataFrame({
    'ID': np.arange(1, 16),
    "PixelID": np.arange(201, 216),
    "Density": [15]*5 + [25]*5 + [35]*5,
    'Type': ['Mixed']*5 + ['Conifer']*5 + ['Decidous']*5,
    "Season": np.arange(2002, 2017),
    'Age': [2, 2, 4, 6, 3, 4, 7, 8, 6, 6, 8, 2, 4, 5, 6],
    'NDVI':  [0.7, 0.5, 0.3, -0.4, 0.0, 0.4, 0.4, 0.5, 0.6, 0.9, 0.7, 0.3, -0.4, 0.0, 0.4],
    'SAVI':  [0.6, -0.8, 0.6, -0.4, 0.3, 0.2, 0.2, 0.4, 1.0, 1.0, -0.5, 0.3, -0.7, 0.4, 0.8],
    'MSAVI': [0.3, 1.0, 0.7, 0.2, -0.7, -0.1, -0.8, 0.6, -0.4, 0.8, -0.8, 0.6, -0.4, 0.3, 0.2],
    'EVI':   [0.6, 0.3, -0.2, 0.6, 1.0, 0.1, 0.4, -0.7, -0.1, 0.6, -0.8, 0.6, -0.4, 0.3, 0.2],
    'EVI2':  [0.4, -0.3, 0.9, -0.9, 0.3, -0.4, 1.0, -0.5, 0.3, 0.8, -0.8, 0.6, -0.4, 0.3, 0.2],
    'NDWI':  [-0.6, 0.3, 0.7, 0.4, -0.1, 0.2, 0.2, 0.2, 0.6, -0.4, 0.4, 0.6, -0.8, 0.1, 0.8],
    'NBR':   [1.0, -0.5, 0.3, -0.7, 0.4, 0.2, -0.1, 0.5, 0.6, 0.4, -0.7, -0.1, 2.6, 0.7, -0.3],
    'TCB':   [-0.4, 0.4, 0.6, -0.8, 0.1, 0.8, 0.4, -0.3, 0.9, 0.4, -0.7, -0.1, 2.6, 0.7, -0.3],
    'TCG':   [0.7, -0.1, 0.5, 0.6, -0.2, 0.3, -0.1, 0.2, 0.2, 0.4, -0.7, -0.1, 2.6, 0.7, -0.3],
    'TCW':   [0.4, -0.7, -0.1, 2.6, 0.7, -0.3, 1.0, 0.7, 0.2, -0.1, 2.6, 0.7, -0.3, 1.0, 0.7],
    'target': [1]*6 + [0]*9,
    'SrvvR_Date': pd.date_range(start="2002-01-01", end="2016-01-01", freq="YS")
})
    
    scores = get_test_errors(dummy_model, dummy_data, dummy_test)
    expected_keys = {
        'F1 Score', 'F2 Score', 'Precision', 'Recall', 'Accuracy',
        'AUC', 'AP', '% Low Risk', '% High Risk'
    }
    assert expected_keys.issubset(scores.keys())
    assert all(isinstance(v, float) for v in scores.values())
    
def test_get_error_metrics_values():
    '''
    Test to ensure error metrics are correct.
    '''
    
    y_pred = np.array([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1])
    y_prob = np.array(
        [0.1, 0.2, 0.4, 0.4, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 
        0.1, 0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9,0.9])
    y_true = np.array([0,0,1,1,1,1,0,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1])

    scores = get_error_metrics(y_pred, y_prob, y_true)
    
    # compute prediction cases according to 0 as positive label
    TP = 6; TN = 7; FP = 5; FN = 4
    
    expected_precision = TP / (TP + FP)
    expected_recall = TP / (TP + FN)
    expected_accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    expected_f2 = 5 * expected_precision * expected_recall / (4 * expected_precision + expected_recall)
    
    expected_pct_low = 100 * round(10 / 22,3)
    expected_pct_high = 100 * round(12 / 22,3)
    
    assert round(expected_precision,3) == scores['Precision']
    assert round(expected_recall,3) == scores['Recall']
    assert round(expected_accuracy,3) == scores['Accuracy']
    assert round(expected_f1,3) == scores['F1 Score']
    assert round(expected_f2,3) == scores['F2 Score']
    assert expected_pct_low == scores['% Low Risk']
    assert expected_pct_high == scores['% High Risk']