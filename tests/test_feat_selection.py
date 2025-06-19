from sklearn.utils.validation import check_is_fitted
from sklearn import set_config
from sklearn.utils import Bunch
import shap
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pytest
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.feat_selection import ImportanceFeatureSelector

# Suppress expected warnings for tests
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress shap DeprecationWarning about missing __sklearn_tags__
warnings.filterwarnings(
    "ignore",
    message=".*__sklearn_tags__.*",
    category=DeprecationWarning
)

# Suppress convergence warnings from LogisticRegression
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)

# Test data 
test_X = pd.DataFrame(
        {
            'ID': [1]*4 + [2]*8,
            'PixelID': ['1_1']*4 + ['2_1']*4 + ['2_2']*4,
            'Density': [1434.54]*4 + [2124.3]*8,
            'Type':  ['Conifer']*4 + ['Mixed']*4 + ['Decidous']*4,
            'Season': [2008]*12,
            'Age': [2, 4, 6, 7]*3,
            'SrvvR_Date': [1]*12,
            'NDVI':  [0.1]*12,
            'SAVI':  [0.1]*12,
            'MSAVI': [0.1]*12,
            'EVI':   [0.1]*12,
            'EVI2':  [0.1]*12,
            'NDWI':  [0.1]*12,
            'NBR':   [0.1]*12,
            'TCB':   [0.1]*12,
            'TCG':   [0.1]*12,
            'TCW':   [0.1]*12
        }
    )

test_y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1])



# Exception Handling Tests
def test_invalid_method_input():
    """
    Test if value error is raised when user inputs invalid value for "method".
    """
    with pytest.raises(ValueError, match='Please choose a feature selection method: "SHAP" or "permute".'):
        ImportanceFeatureSelector(RandomForestClassifier())
    with pytest.raises(ValueError, match='Feature selection method should be either "SHAP" or "permute".'):
        ImportanceFeatureSelector(RandomForestClassifier(), 'shap')


def test_invalid_features():
    """
    Test if value error is raised when "drop_features" and "keep_features" are not lists.
    """
    # Drop Features
    with pytest.raises(ValueError, match=r'List expected for "drop_features", got *'):
        ImportanceFeatureSelector(
            RandomForestClassifier(), 'SHAP', drop_features='Age')
    
    # Keep Features
    with pytest.raises(ValueError, match=r'List expected for "keep_features", got *'):
        ImportanceFeatureSelector(
            RandomForestClassifier(), 'SHAP', keep_features='Age')


def test_invalid_scaler_input():
    """
    Test if value error is raised when "scaler" is not boolean.
    """
    with pytest.raises(ValueError, match=r'"scaler" expectes an bool, got *'):
        ImportanceFeatureSelector(
            RandomForestClassifier(), 'SHAP', scaler='True')
        

def test_invalid_num_feats():
    """
    Test if value error is raised when "num_feats" is not an integer > 0.
    """
    # Not an integer
    with pytest.raises(ValueError, match=r'"num_feats" expectes an integer, got *'):
        ImportanceFeatureSelector(
            RandomForestClassifier(), 'SHAP', num_feats=1.1)
    # Value < 0
    with pytest.raises(ValueError, match=r'"num_feats" expects an integer > 0, got *'):
        ImportanceFeatureSelector(
            RandomForestClassifier(), 'SHAP', num_feats=-1)




# Initilization Test
def test_initialization_defaults():
    """
    Test if the selector is initilized with the default values.
    """
    pipeline = ImportanceFeatureSelector(RandomForestClassifier(), 'SHAP')
    assert pipeline.num_feats == 5
    assert pipeline.drop_features == []
    assert pipeline.keep_features == []
    assert pipeline.scaler is False
    assert pipeline.selected_features is None




# Model compatibility: Test if the fit, predict and score methods works for all three models (RF, GBM and LR).
def test_fit_shap_rf():
    """
    Test if function runs for RandomForestClassifier with SHAP feature selection.
    """
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'SHAP')
    pipeline.fit(test_X, test_y)
    check_is_fitted(pipeline.estimator)
    check_is_fitted(pipeline.preprocessor_)
    assert isinstance(pipeline.plot_data, pd.Series)
    assert len(pipeline.selected_features) == 5
    assert isinstance(pipeline.values, shap._explanation.Explanation)
    assert len(pipeline.predict(test_X)) == len(test_y)
    assert isinstance(pipeline.score(test_X, test_y), float)


def test_permute_rf():
    """
    Test if the function runs for RandomForestClassifier with permutation feature selection.
    """
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'permute')
    pipeline.fit(test_X, test_y)
    check_is_fitted(pipeline.estimator)
    check_is_fitted(pipeline.preprocessor_)
    assert isinstance(pipeline.plot_data, pd.Series)
    assert len(pipeline.selected_features) == 5
    assert isinstance(pipeline.values, Bunch)
    assert len(pipeline.predict(test_X)) == len(test_y)
    assert isinstance(pipeline.score(test_X, test_y), float)


def test_fit_shap_gbm():
    """
    Test if function runs for XGBClassifier with SHAP feature selection.
    """
    pipeline = ImportanceFeatureSelector(
        XGBClassifier(random_state=591, n_jobs=-1), 'SHAP')
    pipeline.fit(test_X, test_y)
    check_is_fitted(pipeline.estimator)
    check_is_fitted(pipeline.preprocessor_)
    assert isinstance(pipeline.plot_data, pd.Series)
    assert len(pipeline.selected_features) == 5
    assert isinstance(pipeline.values, shap._explanation.Explanation)
    assert len(pipeline.predict(test_X)) == len(test_y)
    assert isinstance(pipeline.score(test_X, test_y), float)


def test_fit_permute_gbm():
    """
    Test if function runs for XGBClassifier with permutation feature selection.
    """
    pipeline = ImportanceFeatureSelector(
        XGBClassifier(random_state=591, n_jobs=-1), 'permute')
    pipeline.fit(test_X, test_y)
    check_is_fitted(pipeline.estimator)
    check_is_fitted(pipeline.preprocessor_)
    assert isinstance(pipeline.plot_data, pd.Series)
    assert len(pipeline.selected_features) == 5
    assert isinstance(pipeline.values, Bunch)
    assert len(pipeline.predict(test_X)) == len(test_y)
    assert isinstance(pipeline.score(test_X, test_y), float)


def test_fit_shap_lr():
    """
    Test if function runs for LogisticRegression with SHAP feature selection.
    """
    pipeline = ImportanceFeatureSelector(
        LogisticRegression(max_iter=500), 'SHAP')
    pipeline.fit(test_X, test_y)
    check_is_fitted(pipeline.estimator)
    check_is_fitted(pipeline.preprocessor_)
    assert isinstance(pipeline.plot_data, pd.Series)
    assert len(pipeline.selected_features) == 5
    assert isinstance(pipeline.values, shap._explanation.Explanation)
    assert len(pipeline.predict(test_X)) == len(test_y)
    assert isinstance(pipeline.score(test_X, test_y), float)


def test_fit_permute_lr():
    """
    Test if function runs for LogisticRegression with permutation feature selection.
    """
    pipeline = ImportanceFeatureSelector(
        LogisticRegression(max_iter=500), 'permute')
    pipeline.fit(test_X, test_y)
    check_is_fitted(pipeline.estimator)
    check_is_fitted(pipeline.preprocessor_)
    assert isinstance(pipeline.plot_data, pd.Series)
    assert len(pipeline.selected_features) == 5
    assert isinstance(pipeline.values, Bunch)
    assert len(pipeline.predict(test_X)) == len(test_y)
    assert isinstance(pipeline.score(test_X, test_y), float)




# Functionality Tests
def test_scaler():
    """
    Test if StandardScaler is applied if scaler=True.
    """
    pipeline = ImportanceFeatureSelector(
        LogisticRegression(), 'permute', scaler=True)
    pipeline.fit(test_X, test_y)
    assert 'standardscaler' in pipeline.preprocessor_.named_transformers_.keys()


def test_get_feature_name():
    """
    Test if get_feature_name() method extracts the feature names used in initial fitting correctly. 
    """
    exp_feature_names = ['Type_Conifer', 'Type_Decidous', 'Type_Mixed', 'Density', 'Age',
                         'NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR', 'TCB', 'TCG',
                         'TCW']
    pipeline = ImportanceFeatureSelector(
        LogisticRegression(), 'permute', scaler=True)
    pipeline.fit(test_X, test_y)
    assert pipeline.get_feature_names() == exp_feature_names


def test_drop_feature():
    """
    Test if the correct columns are dropped when specified.
    """
    exp_drop_cols = {'ID', 'PixelID', 'Season', 'SrvvR_Date', 'Age', 'NDVI'}
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(n_jobs=-1), 'permute', drop_features=['Age', 'NDVI'])
    pipeline.fit(test_X, test_y)
    assert exp_drop_cols.difference(
        set(pipeline.plot_data.index)) == exp_drop_cols


def test_keep_feature():
    """
    Test if the correct columns are kept when specified.
    """
    keep_cols = ['NDVI', 'TCB']
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'permute', num_feats=3, keep_features=keep_cols)
    pipeline.fit(test_X, test_y)
    assert all(k in pipeline.selected_features for k in keep_cols)
    assert len(pipeline.selected_features) == 3 + len(keep_cols)


def test_num_feats():
    """
    Test if the function selects the specified number of features.
    """
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'permute', num_feats=3)
    pipeline.fit(test_X, test_y)
    assert len(pipeline.selected_features) == 3


def test_transform():
    """
    Test if the preprocessor is updated after refitting the model with the selected features.
    """
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'SHAP', num_feats=3)
    pipeline.fit(test_X, test_y)
    assert pipeline.transform(test_X).shape[1] == 3


def test_predict_proba():
    """
    Test if predict_proba predicts the class probability for X.
    """
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'SHAP')
    pipeline.fit(test_X, test_y)
    assert pipeline.predict_proba(test_X).shape == (len(test_y), 2)


def test_predict_log_proba():
    """
    Test if predict_proba predicts the class log-probability for X.
    """
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'SHAP')
    pipeline.fit(test_X, test_y)
    assert pipeline.predict_log_proba(test_X).shape == (len(test_y), 2)


def test_get_params():
    """
    Test if get_params() returns the hyperparameters for the fitted model.
    """
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'SHAP')
    pipeline.fit(test_X, test_y)
    assert isinstance(pipeline.get_params(), dict)
    assert pipeline.get_params() == pipeline.estimator.get_params()




# MetaData Routing Tests
def test_fit_routing():
    """
    Test if the additional fit parameters are correctly routed.
    """
    set_config(enable_metadata_routing=True)
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'SHAP')
    pipeline.estimator.set_fit_request(sample_weight=True)
    pipeline.fit(test_X, test_y, sample_weight=np.ones(12))
    pipeline.get_metadata_routing().validate_metadata(
        method='fit', params={'sample_weight': np.ones(12)})
    set_config(enable_metadata_routing=False)


def test_predict_routing():
    """
    Test if the additional predict parameters are correctly routed.
    """
    set_config(enable_metadata_routing=True)
    pipeline = ImportanceFeatureSelector(
        XGBClassifier(random_state=591, n_jobs=-1), 'SHAP')
    pipeline.fit(test_X, test_y)
    pipeline.estimator.set_predict_request(output_margin=True)
    pipeline.predict(test_X, output_margin=True)
    pipeline.get_metadata_routing().validate_metadata(
        method='predict', params={'output_margin': True})
    set_config(enable_metadata_routing=False)


def test_score_routing():
    """
    Test if the additional score parameters are correctly routed.
    """
    set_config(enable_metadata_routing=True)
    pipeline = ImportanceFeatureSelector(
        RandomForestClassifier(random_state=591, n_jobs=-1), 'SHAP')
    pipeline.fit(test_X, test_y)
    pipeline.estimator.set_score_request(sample_weight=True)
    pipeline.score(test_X, test_y, sample_weight=np.ones(12))
    pipeline.get_metadata_routing().validate_metadata(
        method='score', params={'sample_weight': np.ones(12)})
    set_config(enable_metadata_routing=False)





