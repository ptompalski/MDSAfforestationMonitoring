import shap
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.base import MetaEstimatorMixin, TransformerMixin, clone
from sklearn.utils._metadata_requests import MetadataRouter, MethodMapping, process_routing
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _estimator_has, check_is_fitted


class SHAPFeatureSelector(MetaEstimatorMixin, TransformerMixin):
    """Feature Selection with SHAP Global Feature Importance.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), SHAPFeatureSelector will select a specified 
    number of features with the highest mean absolute SHAP value. 

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A sklearn estimator.

    num_feats : int, default=5
        The number of features to select. 

    drop_features : list of str, default=None
        Features to exclude from the model.

    keep_features : list of str, default=None
        Features to keep during feature selection.

    scaler : bool, default=False
        Whether to apply StandardScaler to numeric columns.    


    Attributes
    ----------
    estimator : ``Estimator`` instance
        The fitted estimator.

    selected_features : list of str
        List of features used to fit the final model (including keep_features).

    preprocessor_ : ColumnTransformer
        The fitted preprocessor for X.

    values : pd.Series
        Global SHAP value for the selected features.
    """

    def __init__(self, estimator, num_feats=5, drop_features=None, keep_features=None, scaler=False):
        self.estimator = estimator
        self.num_feats = num_feats
        self.drop_features = drop_features
        self.keep_features = keep_features
        self.scaler = scaler
        self.selected_features = None

