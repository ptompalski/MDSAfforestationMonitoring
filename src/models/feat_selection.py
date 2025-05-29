import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.base import MetaEstimatorMixin, TransformerMixin, clone
from sklearn.utils._metadata_requests import MetadataRouter, MethodMapping, process_routing, _routing_enabled
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _estimator_has, check_is_fitted
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
from sklearn.linear_model import LogisticRegression
import shap 
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import click
from pathlib import Path
import joblib

class ImportanceFeatureSelector(MetaEstimatorMixin, TransformerMixin):
    """Feature Selection with SHAP Global Feature Importance.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), ImportanceFeatureSelector will select a specified 
    number of features with the highest mean absolute SHAP value/Permutation Importance. 

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A sklearn estimator.

    num_feats : int, default=5
        The number of features to select. 

    drop_features : list of str, default=[]
        Features to exclude from the model.

    keep_features : list of str, default=[]
        Features to keep during feature selection.

    scaler : bool, default=False
        Whether to apply StandardScaler to numeric columns.    

    method : str, {'SHAP', 'permute'}
        Method to evaluate feature importance.
        - 'SHAP': Uses SHAP (SHapley Additive exPlanations) to compute feature importance.
        - 'permute': Uses permutation importance (`sklearn.inspection.permutation_importance`) to compute feature importance.

    Attributes
    ----------
    estimator : ``Estimator`` instance
        The fitted estimator.

    selected_features : list of str
        List of features used to fit the final model (including keep_features).

    preprocessor_ : ColumnTransformer
        The fitted preprocessor for X.

    values : dict 
        Dictionary-like object with the following attributes:
        - If method = 'SHAP',
            values : ndarray of shape (n_samples, n_features)
                SHAP values for each sample.
            base_values : ndarray of shape (n_classes,)
                Expected value for the model output (proportion for each class).
            data : ndarray of shape (n_samples, n_features)
                The input data corresponding to the SHAP values. 
        - If method = 'permute',
            importances_mean : ndarray of shape (n_features, )
                Mean feature importance over 5 repeats.
            importances_std : ndarray of shape (n_features, )
                Standard deviation over 5 repeats.
            importances : ndarray of shape (n_features, 5)
                Raw permutation importance scores for each repeat.

    plot_data : pd.Series
         Feature importance scores indexed by feature name. Used as input for plotting feature importance.

    """

    def __init__(self, estimator, method=None, num_feats=5, drop_features=[], keep_features=[], scaler=False):
        
        # Exception Handling
        if method == None:
            raise ValueError('Please choose a feature selection method: "SHAP" or "permute".')
        if method not in ['SHAP', 'permute']:
            raise ValueError(
                'Feature selection method should be either "SHAP" or "permute".')
        if not isinstance(drop_features, list):
            raise ValueError(
                f'List expected for "drop_features", got {type(drop_features)}')
        if not isinstance(keep_features, list):
            raise ValueError(
                f'List expected for "keep_features", got {type(keep_features)}')
        if not isinstance(num_feats, int):
            raise ValueError(
                f'"num_feats" expectes an integer, got {type(num_feats)}')
        if num_feats <= 0:
            raise ValueError(
                f'"num_feats" expects an integer > 0, got {num_feats}'
            )
        if not isinstance(scaler, bool):
            raise ValueError(
                f'"scaler" expectes an bool, got {type(scaler)}')
        self.estimator = estimator
        self.num_feats = num_feats
        self.drop_features = drop_features
        self.keep_features = keep_features
        self.scaler = scaler
        self.selected_features = None
        self.method = method
        
    @property
    def feature_importances_(self):
        """Feature importance of the features.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
        """
        check_is_fitted(self)
        return self.estimator.feature_importances_

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

    def selector(self):
        """Get mean feature importance and select features with highest importances.
        """
        if self.method == 'SHAP':
            self.plot_data = pd.DataFrame(
                self.values.values,
                columns=self.get_feature_names()
            ).abs().mean()

        if self.method == 'permute':
            self.plot_data = pd.Series(
                self.values.importances_mean,
                index=self.get_feature_names()
            )

        self.selected_features = self.plot_data.sort_values(
            ascending=False).drop(index=self.keep_features).head(
            self.num_feats).index.tolist() + self.keep_features

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

    def fit(self, X, y, **fit_params):
        """
        Fit the estimator and conduct feature selection through evaluating SHAP global feature importance.
        The estimator is then refitted with the selected features.

        This method performs the following steps:
        1. Preprocess the training data (X).
        2. Fit the estimator with the preprocessed data.
        3. Calculate feature importance for each feature.
        4. Select features with the importance to refit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of observations and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target feature.

        **fit_params : dict
            Additional parameters to pass to the ``fit`` method of the estimator.
            Call `estimator.set_fit_request()` prior to fitting for metadata routing.

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
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = Bunch(estimator=Bunch(fit=fit_params))
        estimator = clone(self.estimator)
        estimator.fit(X_enc, y, **routed_params.estimator.fit)

        if self.method == 'SHAP':
            # SHAP Explanation
            if isinstance(estimator, LogisticRegression):
                explainer = shap.Explainer(
                    estimator, X_enc, feature_names=self.get_feature_names())
                shap_values = explainer.shap_values(X_enc)
            else:
                explainer = shap.Explainer(estimator)
                shap_values = explainer.shap_values(X_enc, approximate=True)
            self.values = shap.Explanation(
                (shap_values[:, :, 0]
                if shap_values.ndim == 3 else shap_values),
                data=np.array(X_enc),
                base_values=explainer.expected_value,
                feature_names=self.get_feature_names()
            )
            self.selector()

        if self.method == 'permute':
            # Permutation Importance
            self.values = permutation_importance(estimator, X_enc, y)
            self.selector()

        # Refit estimator with selected features
        return self.estimator.fit(X_enc[self.selected_features], y,  **routed_params.estimator.fit)

    @available_if(_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Reduce X to the selected features and predict using the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        **predict_params : dict
            Additional parameters to pass to the ``predict`` method of the
            underlying estimator. Call `estimator.set_predict_request()` prior to fitting for metadata routing.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self)
        if _routing_enabled():
            routed_params = process_routing(self, "predict", **predict_params)
        else:
            routed_params = Bunch(estimator=Bunch(predict=predict_params))

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
            Additional parameters to pass to the ``score`` method of the underlying estimator.
             Call `estimator.set_score_request()` prior to fitting for metadata routing.

        Returns
        -------
        score : float
            Score of the estimator computed with the selected features.
        """
        check_is_fitted(self)
        if _routing_enabled():
            routed_params = process_routing(self, "score", **score_params)
        else:
            routed_params = Bunch(estimator=Bunch(score=score_params))

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


@click.command()
@click.option(
    '--estimator',
    help='The estimator used to gauge feature importance (Logistic Regression (lr), Random Forest (rf), Gradient Boosting (gbm))',
    type=click.Choice(['lr','gbm','rf']),
    default='lr'
)
@click.option(
    '--method',
    help='Feature selection method to implement',
    type=click.Choice(['SHAP', 'permute']),
    required=True
)
@click.option(
    '--drop_features', 
    type=list, 
    default='',
    help='Comma-separated list of features to drop (e.g., "feat1,feat2")'
)
@click.option(
    '--input_path',
    help='Path to input training data',
    type=click.Path(dir_okay=False),
    required=True
)
@click.option(
    '--output_dir',
    help='Directory to store the fitted `ImportanceFeatureSelector` instance.',
    type=click.Path(file_okay=False),
    required=True
)
@click.option(
    '--random_state',
    help='Random state for reproducibility: May effect random elements of estimators.',
    type=int,
    default=591
)
def main(estimator,method,drop_features,input_path,output_dir,random_state):
    '''
    CLI for SHAP and permutation importance feature selections
    '''
    # prepare data
    train_df = pd.read_parquet(input_path).dropna()
    X = train_df.drop(columns='target'); y = train_df['target']
    
    # standard scale features if given logistic regression model
    scale_num_feats = True if estimator == 'lr' else False
    
    # format drop_features
    drop_features = drop_features.split(',') if drop_features != '' else []
    
    # set up directories:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    
    # set up esimator and output
    if estimator == 'lr':
        output_path = output_dir/f'fitted_logistic_regression_{method.lower()}.joblib'
        estimator = LogisticRegression(random_state=random_state,n_jobs=-1)
    elif estimator == 'gbm':
        output_path = output_dir/f'fitted_gradient_boosting_{method.lower()}.joblib'
        estimator = XGBClassifier(random_state=random_state,n_jobs=-1)
    else:
        output_path = output_dir/f'fitted_random_forest_{method.lower()}.joblib'
        estimator = RandomForestClassifier(random_state=random_state,n_jobs=-1)
    
    click.echo('Fitting feature importance selector...')
    
    # create feature importance selector
    imp_feat_selector = ImportanceFeatureSelector(
        estimator=estimator,
        drop_features=drop_features,
        scaler=scale_num_feats,
        method=method        
    )
    imp_feat_selector.fit(X,y)
    click.echo('Done')
    
    # save model
    joblib.dump(imp_feat_selector,output_path)
    click.echo(f'saved to {output_path}')
    
    
if __name__ == '__main__':
    main()