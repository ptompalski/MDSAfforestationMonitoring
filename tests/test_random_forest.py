import pytest
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.random_forest import build_rf_pipline

@pytest.fixture()
def sample_data():
    sample_data = pd.DataFrame({
    'ID':np.arange(1,10),
    "PixelID":np.arange(101,110),
    "Density": [10]*3 + [20]*3 + [30]*3,
    "Season": np.arange(2001,2010),
    'Type':['Decidous']*3 + ['Mixed']*3 + ['Conifer']*3 ,
    'Age': [1,1,3,5,2,3,6,7,5],
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3, 0.3, 0.6, 0.7],   
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1, 0.1, 0.5, 0.9], 
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2, -0.9, 0.5, -0.5],   
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0, 0.3, -0.8, -0.2],   
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5, 0.9, -0.6, 0.2],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1, 0.1, 0.1, 0.5,],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1, -0.2, 0.4, 0.5],  
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7, 0.3, -0.4, 0.8],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2, -0.2, 0.1, 0.1,],   
    'TCW':   [0.3, -0.8, -0.2, 2.5, 0.6, -0.4, 0.9, 0.6, 0.1,], 
    'target': [0]*4 + [1]*5,
    'SrvvR_Date': pd.date_range(start="2001-01-01", end="2009-01-01", freq="YS"),
    #'DOY': [1]*9
    })
    X = sample_data.drop(columns='target')
    y = sample_data['target']
    groups = sample_data['ID']

    return X,y,groups


@pytest.mark.parametrize("feat_select", [None, "RFE", "RFECV"])
def test_is_pipeline(feat_select):
    '''
    Test that the model is an instance of an sklearn pipeline.
    '''
    pipeline = build_rf_pipline(feat_select=feat_select)
    assert isinstance(pipeline, Pipeline)


def test_invalid_feat_select_raises():
    '''
    Test error-handling when an invalid feature selection method is given.
    '''
    with pytest.raises(ValueError):
        build_rf_pipline(feat_select='INVALID')
    with pytest.raises(ValueError):
        build_rf_pipline(feat_select=0)
    with pytest.raises(ValueError):
        build_rf_pipline(feat_select=[])
    with pytest.raises(ValueError):
        build_rf_pipline(feat_select='rFe')


def test_invalid_drop_features_type():
    '''
    Test error-handling when incorrect type given to drop_features.
    '''
    with pytest.raises(ValueError):
        build_rf_pipline(drop_features="notalist")
    with pytest.raises(ValueError):
        build_rf_pipline(drop_features=7)

@pytest.mark.parametrize(
    "feat_select, drop_features, step_RFE, num_feats_RFE, min_num_feats_RFECV, num_folds_RFECV, scoring_RFECV, random_state, kwargs",
    [
        (None, ['NDVI'], 1, 4, 3, 4, "f1", 123, {"n_estimators": 100, "max_depth": 3}),
        ("RFE", ['EVI', 'NBR'], 2, 2, 3, 3, "precision", 42, {"n_estimators": 50, "max_depth": 5,}),
        ("RFECV", None, 1, 3, 2, 2, "recall", 777, {"n_estimators": 150, "max_depth": 4}),
    ]
)
def test_pipeline_all_parameters(
    feat_select,
    drop_features,
    step_RFE,
    num_feats_RFE,
    min_num_feats_RFECV,
    num_folds_RFECV,
    scoring_RFECV,
    random_state,
    kwargs
):
    '''
    Test to ensure hyperparameters are defined properly in pipeline.
    '''
    pipeline = build_rf_pipline(
        feat_select=feat_select,
        drop_features=drop_features,
        step_RFE=step_RFE,
        num_feats_RFE=num_feats_RFE,
        min_num_feats_RFECV=min_num_feats_RFECV,
        num_folds_RFECV=num_folds_RFECV,
        scoring_RFECV=scoring_RFECV,
        random_state=random_state,
        **kwargs
    )

    # Get model or wrapped estimator
    if feat_select == None:
        model = pipeline.named_steps['randomforestclassifier']
    elif feat_select == "RFE":
        rfe = pipeline.named_steps['rfe']
        model = rfe.estimator
        assert rfe.n_features_to_select == num_feats_RFE
        assert rfe.step == step_RFE
    else:  # RFECV
        rfecv = pipeline.named_steps['rfecv']
        model = rfecv.estimator
        assert rfecv.min_features_to_select == min_num_feats_RFECV
        assert rfecv.scoring == scoring_RFECV
        assert isinstance(rfecv.cv, GroupKFold)
        assert rfecv.cv.n_splits == num_folds_RFECV

    # Check model hyperparameters
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == kwargs['n_estimators']
    assert model.max_depth == kwargs['max_depth']
    assert model.random_state == random_state

    # Check preprocessor drop column setup
    preprocessor = pipeline.named_steps['columntransformer']
    dropped = preprocessor.transformers[0][2]
    expected_drops = ['ID', 'PixelID', 'Season', 'SrvvR_Date'] + (drop_features if drop_features else [])
    assert sorted(dropped) == sorted(expected_drops)

def test_pipeline_fit_no_feat_select(sample_data):
    '''
    Test to ensure model can be fitted with no feature selection.
    '''
    X, y, _ = sample_data
    pipeline = build_rf_pipline()
    pipeline.fit(X, y)
    check_is_fitted(pipeline)  

def test_pipeline_fit_with_rfe(sample_data):
    '''
    Test to ensure model can be fitted with RFE.
    '''
    X, y, _ = sample_data
    pipeline = build_rf_pipline(feat_select='RFE', num_feats_RFE=2)
    pipeline.fit(X, y)
    check_is_fitted(pipeline)

def test_pipeline_fit_with_rfecv(sample_data):
    '''
    Test to ensure model can be fitted with RFECV.
    '''
    X, y, groups = sample_data
    pipeline = build_rf_pipline(feat_select='RFECV', num_folds_RFECV=2)
    pipeline.fit(X, y,rfecv__groups=groups)
    check_is_fitted(pipeline)

def test_rfecv_value_error_no_groups(sample_data):
    '''
    Test to ensure an error is thrown when fitting RFECV pipeline with no groups
    '''
    with pytest.raises(ValueError):
        X, y, groups = sample_data
        pipeline = build_rf_pipline(feat_select='RFECV', num_feats_RFE=2)
        pipeline.fit(X, y)
