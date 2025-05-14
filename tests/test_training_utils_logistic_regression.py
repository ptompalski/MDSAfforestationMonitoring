import pytest
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from src.model.logistic_regression import build_logreg_pipeline
from src.model.training_utils_logistic_regression import cross_validation_wrapper

@pytest.fixture()
def sample_data():
    '''
    Create sample dataframe for test fitting logistic regression via CV.
    '''
    sample_data = pd.DataFrame({
        'ID':       np.arange(1, 16),
        'PixelID':  np.arange(101, 116),
        'Type':     ['Decidous']*5 + ['Mixed']*5 + ['Conifer']*5,
        'NDVI':     [0.8, 0.6, 0.2, -0.5, -0.1, 0.3, 0.3, 0.6, 0.7, 0.8, 0.6, 0.2, -0.5, -0.1, 0.3],
        'SAVI':     [0.7, -0.9, 0.5, -0.5, 0.2, 0.1, 0.1, 0.5, 0.9, 0.9, -0.6, 0.2, -0.8, 0.3, 0.7],
        'MSAVI':    [0.4, 0.9, 0.6, 0.1, -0.8, -0.2, -0.9, 0.5, -0.5, 0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
        'EVI':      [0.5, 0.4, -0.3, 0.7, 0.9, 0.0, 0.3, -0.8, -0.2, 0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
        'EVI2':     [0.3, -0.4, 0.8, -1.0, 0.2, -0.5, 0.9, -0.6, 0.2, 0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
        'NDWI':     [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1, 0.1, 0.1, 0.5, -0.5, 0.3, 0.5, -0.9, 0.0, 0.7],
        'NBR':      [0.9, -0.6, 0.2, -0.8, 0.3, 0.1, -0.2, 0.4, 0.5, 0.3, -0.8, -0.2, 2.5, 0.6, -0.4],
        'TCB':      [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7, 0.3, -0.4, 0.8, 0.3, -0.8, -0.2, 2.5, 0.6, -0.4],
        'TCG':      [0.6, -0.2, 0.4, 0.5, -0.3, 0.2, -0.2, 0.1, 0.1, 0.3, -0.8, -0.2, 2.5, 0.6, -0.4],
        'TCW':      [0.3, -0.8, -0.2, 2.5, 0.6, -0.4, 0.9, 0.6, 0.1, -0.2, 2.5, 0.6, -0.4, 0.9, 0.6],
        'Density':  [800]*15,
        'target':   [0]*8 + [1]*7
    })
    return sample_data

@pytest.fixture()
def simple_param_grid():
    return {
        'logisticregression__C': [0.1, 1.0],
        'logisticregression__penalty': ['l2']
    }

def test_basic_random_search_runs(sample_data, simple_param_grid):
    '''
    Test to ensure cross-validation wrapper completes and returns expected results.
    '''
    result = cross_validation_wrapper(
        model_pipeline=build_logreg_pipeline(),
        df=sample_data,
        param_grid=simple_param_grid,
        method='random',
        num_iter=2,
        num_folds=2,
        random_state=42
    )
    assert 'best_model' in result
    assert 'best_score' in result
    assert 'best_params' in result

def test_return_results_flag(sample_data, simple_param_grid):
    '''
    Test to ensure a DataFrame results is returned when requested.
    '''
    result = cross_validation_wrapper(
        model_pipeline=build_logreg_pipeline(),
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
    Test to ensure grid search returns a best parameter.
    '''
    result = cross_validation_wrapper(
        model_pipeline=build_logreg_pipeline(),
        df=sample_data,
        param_grid=simple_param_grid,
        method='grid',
        num_folds=2
    )
    assert result['best_params'] is not None

def test_invalid_method_raises(sample_data, simple_param_grid):
    '''
    Test to ensure error raised when invalid method is passed.
    '''
    with pytest.raises(ValueError):
        cross_validation_wrapper(
            model_pipeline=build_logreg_pipeline(),
            df=sample_data,
            param_grid=simple_param_grid,
            method='banana'
        )

def test_random_state_reproducibility(sample_data, simple_param_grid):
    '''
    Test reproducibility: same seed yields same results.
    '''
    out1 = cross_validation_wrapper(
        model_pipeline=build_logreg_pipeline(random_state=123),
        df=sample_data,
        param_grid=simple_param_grid,
        method='random',
        num_iter=2,
        num_folds=2,
        random_state=123
    )
    out2 = cross_validation_wrapper(
        model_pipeline=build_logreg_pipeline(random_state=123),
        df=sample_data,
        param_grid=simple_param_grid,
        method='random',
        num_iter=2,
        num_folds=2,
        random_state=123
    )
    assert out1['best_score'] == out2['best_score']
    assert out1['best_params'] == out2['best_params']

def test_random_search_allows_scipy_distributions(sample_data):
    '''
    Ensure scipy.stats distributions are accepted by random search.
    '''
    param_dist = {
        'logisticregression__C': randint(1, 3),
        'logisticregression__tol': uniform(1e-4, 1e-3)
    }
    result = cross_validation_wrapper(
        model_pipeline=build_logreg_pipeline(random_state=42),
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
    Ensure GridSearchCV raises TypeError with scipy distributions in grid mode.
    '''
    param_dist = {'logisticregression__C': randint(1, 3)}
    with pytest.raises(TypeError):
        cross_validation_wrapper(
            model_pipeline=build_logreg_pipeline(random_state=42),
            df=sample_data,
            param_grid=param_dist,
            method='grid',
            num_folds=2
        )