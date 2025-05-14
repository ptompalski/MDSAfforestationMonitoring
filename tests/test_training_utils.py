import pytest
import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import randint, uniform
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.gradient_boosting import build_gbm_pipeline
from src.models.training_utils import cross_validation_wrapper

@pytest.fixture()
def sample_data():
    '''
    Create sample dataframe for test fitting to models.
    '''
    sample_data = pd.DataFrame({
        'ID': np.arange(1, 16),
        "PixelID": np.arange(101, 116),
        "Density": [10]*5 + [20]*5 + [30]*5,
        'Type': ['Decidous']*5 + ['Mixed']*5 + ['Conifer']*5,
        "Season": np.arange(2001,2016),
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
        'SrvvR_Date': pd.date_range(start="2001-01-01", end="2015-01-01", freq="YS")
    })
    return sample_data
    
@pytest.fixture()
def simple_param_grid():
    return {
    'xgbclassifier__n_estimators': [1,10],
    'xgbclassifier__learning_rate': [0.001,10],
    'xgbclassifier__max_depth':[1,2]
    }

def test_basic_random_search_runs(sample_data, simple_param_grid):
    '''
    Test to ensure cross-validation wrapper completes and returns expected results
    '''
    result = cross_validation_wrapper(
        model_pipeline=build_gbm_pipeline(),
        df=sample_data,
        param_grid=simple_param_grid,
        method='random',
        num_iter=2,
        num_folds=2
    )
    assert 'best_model' in result
    assert 'best_score' in result
    assert 'best_params' in result
    
def test_return_results_flag(sample_data, simple_param_grid):
    '''
    Test to ensure expected results are returned when return_results is True.
    '''
    result = cross_validation_wrapper(
        model_pipeline=build_gbm_pipeline(),
        df=sample_data,
        param_grid=simple_param_grid,
        method='random',
        num_iter=2,
        num_folds=2,
        return_results=True
    )
    assert isinstance(result['results'], pd.DataFrame)
    assert 'mean_test_score' in result['results'].columns

def test_grid_search_behavior(sample_data, simple_param_grid):
    '''
    Test to ensure grid search returns a best parameter
    '''
    result = cross_validation_wrapper(
        model_pipeline=build_gbm_pipeline(),
        df=sample_data,
        param_grid=simple_param_grid,
        method='grid',
        num_folds=2
    )
    assert result['best_params'] is not None

def test_invalid_method_raises(sample_data, simple_param_grid):
    '''
    Test to ensure error raised when an invalid method is passed.
    '''
    with pytest.raises(ValueError):
        cross_validation_wrapper(
            model_pipeline=build_gbm_pipeline(),
            df=sample_data,
            param_grid=simple_param_grid,
            method='banana'
        )
        
def test_random_state_reprducibility(sample_data, simple_param_grid):
    '''
    Test to ensure reproducibility: CV sessions with same seed give the same result.
    '''
    out1 = cross_validation_wrapper(
        build_gbm_pipeline(random_state=123),
        sample_data, 
        simple_param_grid, 
        num_iter=2,
        num_folds=2,
        random_state=123,
    )
    out2 = cross_validation_wrapper(
        build_gbm_pipeline(random_state=123),
        sample_data, 
        simple_param_grid, 
        num_iter=2,
        num_folds=2,
        random_state=123,
    )
    # note that dataframe is never perfectly equal due to OS level threading assignments
    assert out1['best_score'] == out2['best_score']
    assert out1['best_params'] == out2['best_params']
    
def test_random_search_allows_scipy_distributions(sample_data):
    '''
    Ensure that scipy.stats distributions are accepted by random search
    '''
    param_dist = {
        'xgbclassifier__n_estimators': randint(10, 50),
        'xgbclassifier__learning_rate': uniform(0.01, 0.2),
    }

    result = cross_validation_wrapper(
        model_pipeline=build_gbm_pipeline(random_state=42),
        df=sample_data,
        param_grid=param_dist,
        method='random',
        num_iter=2,
        num_folds=2,
        random_state=42,
        return_results=True
    )

    assert isinstance(result['best_score'], float)
    assert isinstance(result['best_params'], dict)

def test_grid_search_rejects_scipy_distributions(sample_data):
    '''
    Ensure that GridSearchCV raises a ValueError when given scipy.stats distributions.
    '''
    param_dist = {
        'xgbclassifier__n_estimators': randint(10, 50)
    }

    with pytest.raises(TypeError):
        cross_validation_wrapper(
            model_pipeline=build_gbm_pipeline(random_state=42),
            df=sample_data,
            param_grid=param_dist,
            method='grid',
            num_folds=2
        )
