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

    @property
    def feature_importances_(self):
        """Feature importance of the features.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
        """
        check_is_fitted(self)
        return self.estimator.feature_importances_()

    @property
    def classes_(self):
        """Class labels for the target feature.

        Returns
        -------
        ndarray of shape (n_classes,)
        """
        return self.estimator.classes_

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

    @available_if(_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Reduce X to the selected features and predict using the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        **predict_params : dict
            Parameters to route to the ``predict`` method of the
            underlying estimator.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self)
        routed_params = process_routing(self, "predict", **predict_params)
        return self.estimator.predict(
            self.transform(X), **routed_params.estimator.predict
        )

    @available_if(_estimator_has("score"))
    def score(self, X, y, **score_params):
        """Reduce X to the selected features and return the score of the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.

        **score_params : dict

        Returns
        -------
        score : float
            Score of the estimator computed with the selected features.
        """
        check_is_fitted(self)
        routed_params = process_routing(self, "score", **score_params)
        return self.estimator.score(
            self.transform(X), y, **routed_params.estimator.score
        )

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples. 

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        return self.estimator.predict_proba(self.transform(X))

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. 
        """
        check_is_fitted(self)
        return self.estimator.predict_log_proba(self.transform(X))

    @available_if(_estimator_has("get_params"))
    def get_params(self):
        """Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        check_is_fitted(self)
        return self.estimator.get_params()

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Returns
        -------
        routing : MetadataRouter
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="score", callee="score"),
        )
        return router
