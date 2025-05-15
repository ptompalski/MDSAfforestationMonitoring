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

    def get_feature_names(self):
        """Get feature names from the preprocessor.

        Returns
        --------
        List of feature names.
        """
        return (self.preprocessor_.named_transformers_['onehotencoder'].get_feature_names_out().tolist() +
                self.preprocessor_.named_transformers_[('standardscaler' if self.scaler else 'remainder')].get_feature_names_out().tolist())

    def transform(self, X):
        """Transformer for training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of observations and
            `n_features` is the number of features.

        Returns 
        --------
        X : array-like of shape (n_samples, n_features) or (n_samples, len(selected_features)) 
            Transformed training data.
        """
        X = pd.DataFrame(
            self.preprocessor_.transform(X),
            columns=self.get_feature_names(),
            index=X.index
        )
        return X[:] if self.selected_features is None else X[self.selected_features]

    def fit(self, X, y, **fit_params):
        """
        Fit the estimator and conduct feature selection through evaluating SHAP global feature importance.
        The estimator is then refitted with the selected features.

        This method performs the following steps:
        1. Preprocess the training data (X).
        2. Fit the estimator with the preprocessed data.
        3. Pass the fitted estimator to SHAP explainer and calculate the mean absolute SHAP values for each feature.
        4. Select features with the highest SHAP values to refit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of observations and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target feature.

        **fit_params : dict
            Parameters to passed to the ``fit`` method of the estimator.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        drop_cols = ['ID', 'PixelID', 'Season', 'SrvvR_Date'] if self.drop_features is None else [
            'ID', 'PixelID', 'Season', 'SrvvR_Date'] + self.drop_features
        categorical_cols = ['Type']
        numeric_cols = X.drop(
            columns=drop_cols + categorical_cols).columns.to_list()

        # Define Preprocessor
        if self.scaler == False:
            self.preprocessor_ = make_column_transformer(
                ('drop', drop_cols),
                (OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                remainder='passthrough',
                force_int_remainder_cols=False,
                verbose_feature_names_out=False
            )
        else:
            self.preprocessor_ = make_column_transformer(
                ('drop', drop_cols),
                (OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                (StandardScaler(), numeric_cols),
                force_int_remainder_cols=False,
                verbose_feature_names_out=False
            )

        # Preprocess training data
        self.preprocessor_.fit(X)
        X_enc = self.transform(X)

        # Fit model
        routed_params = process_routing(self, "fit", **fit_params)
        self.model_ = clone(self.estimator)
        self.model_.fit(X_enc, y, **routed_params.estimator.fit)

        # SHAP Explanation
        explainer = shap.Explainer(self.model_)
        shap_values = explainer.shap_values(X_enc, approximate=True)

        # Select top features
        self.values = pd.DataFrame(
            (shap_values[:, :, 0] if shap_values.ndim == 3 else shap_values),
            index=X_enc.index,
            columns=self.get_feature_names()
        ).abs().mean().sort_values(ascending=False)

        if not (self.keep_features == None):
            self.values = self.values.drop(columns=self.keep_features)
            self.selected_features = self.values.head(
                self.num_feats).index.tolist() + self.keep_features
        else:
            self.selected_features = self.values.head(
                self.num_feats).index.tolist()

        # Refit estimator with selected features
        return self.estimator.fit(X_enc[self.selected_features], y, **routed_params.estimator.fit)
