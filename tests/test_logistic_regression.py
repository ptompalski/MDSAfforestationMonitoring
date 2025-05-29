import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.logistic_regression import build_logreg_pipeline
from src.training.cv_tuning import cross_validation_wrapper


@pytest.fixture()
def sample_data():
    '''
    Create sample dataframe for test fitting to models.
    '''
    sample_data = pd.DataFrame({
        'ID': np.arange(1, 16),
        'PixelID': np.arange(101, 116),
        'Density': [10]*5 + [20]*5 + [30]*5,
        'Type': ['Decidous']*5 + ['Mixed']*5 + ['Conifer']*5,
        'Season': np.arange(2001,2016),
        'Age': [1, 1, 3, 5, 2, 3, 6, 7, 5, 5, 7, 1, 3, 4, 5],
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
        #'DOY': [1]*15,
        'SrvvR_Date': pd.date_range(start="2001-01-01", end="2015-01-01", freq="YS")
    })
    return sample_data


@pytest.mark.parametrize("fs", [None, "RFE", "RFECV"])
def test_is_pipeline(fs):
    """Test that `build_logreg_pipeline` returns a Pipeline for each feature selection option."""

    pipe = build_logreg_pipeline(feat_select=fs)
    assert isinstance(pipe, Pipeline)


def test_invalid_feat_select():
    """Test that providing an invalid feature selection string raises a ValueError."""

    with pytest.raises(ValueError):
        build_logreg_pipeline(feat_select="banana")



def test_pipeline_fits(sample_data):
    """Test that the logistic regression pipeline can be fitted and yields a valid accuracy."""

    pipe = build_logreg_pipeline()
    result = cross_validation_wrapper(
        model_pipeline=pipe,
        df=sample_data,
        param_grid={},         
        method='grid',         
        num_folds=2,           
        scoring='accuracy',    
    )
    trained = result['best_model']
    accuracy = result['best_score']
    check_is_fitted(trained)
    assert 0.0 <= accuracy <= 1.0