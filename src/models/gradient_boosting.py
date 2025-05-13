from xgboost import XGBClassifier
from sklearn.feature_selection import RFE,RFECV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupKFold

def build_gbm_pipeline(
    feat_select: str = None,
    drop_features: list = None,
    step_RFE: int = 1,
    num_feats_RFE: int = 4,
    min_num_feats_RFECV: int = 4,
    num_folds_RFECV: int = 5,
    scoring_RFECV: str = 'f1',
    random_state: int = 591,
    **kwargs
):
    """
    Constructs a Scikit-Learn pipeline for training a Gradient Boosting (XGBoost) model 
    on afforestation data, with optional feature selection and preprocessing.

    Parameters
    ----------
    feat_select : {'RFE', 'RFECV', None}, default=None
        Feature selection method to apply:
        - 'RFE': Recursive Feature Elimination.
        - 'RFECV': Recursive Feature Elimination with Cross-Validation.
        - None: No feature selection is performed.

    drop_features : list of str, optional
        Additional features to drop from the dataset (e.g., indices dropped in prior selection steps). 
        Always drops 'ID' and 'PixelID' by default.

    step_RFE : int, default=1
        Number of features to remove at each iteration of the RFE process.

    num_feats_RFE : int, default=4
        Number of features to retain when using RFE.

    min_num_feats_RFECV : int, default=4
        Minimum number of features to retain when using RFECV.

    num_folds_RFECV : int, default=5
        Number of cross-validation folds to use during RFECV.

    scoring_RFECV : str, default='f1'
        Scoring metric used during RFECV. Must be a valid scoring string recognized by scikit-learn.
        See: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    random_state : int, default=591
        Random state seed for reproducibility.

    **kwargs : dict
        Additional keyword arguments passed to the `XGBClassifier` (e.g., hyperparameters like 
        `n_estimators`, `max_depth`, `learning_rate`, etc.).

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline consisting of preprocessing (column dropping, one-hot encoding) 
        and the XGBoost classifier, optionally wrapped in RFE or RFECV.

    Raises
    ------
    ValueError
        If `feat_select` is not one of {None, 'RFE', 'RFECV'}.
        If `drop_features` is not None or a list.
    """
    # error handling
    if feat_select not in (None, 'RFE', 'RFECV'):
        raise ValueError('feat_select must be one of: {None, \'RFE\', \'RFECV\'}')
        
    if drop_features is not None and not isinstance(drop_features, list):
        raise ValueError('drop_features must be a list or None')
    
    # preprocessor: dropping and one-hot encoding
    drop_cols = (
        ['ID', 'PixelID'] if drop_features == None 
        else ['ID', 'PixelID'] + drop_features
        )
    
    categorical_cols = ['Type']                     

    preprocessor = make_column_transformer(
            ('drop',drop_cols),
            (OneHotEncoder(handle_unknown="ignore"),categorical_cols),
            remainder='passthrough',
            force_int_remainder_cols=False
        )
    
    # XGBoost Classifier model
    xgb_classifier = XGBClassifier(
        eval_metric='logloss',
        random_state=random_state,
        **kwargs
    )
   
    # no feature selection
    if feat_select == None:
        model_pipeline = make_pipeline(
            preprocessor,
            xgb_classifier
        )  
        
    # feature selection with RFE 
    elif feat_select == 'RFE':
        model_pipeline = make_pipeline(
            preprocessor,
            RFE(
                estimator=xgb_classifier,
                n_features_to_select=num_feats_RFE,
                step=step_RFE
            )
        )
      
    # Feature selection with RFECV  
    else:
        model_pipeline = make_pipeline(
            preprocessor,
            RFECV(
                estimator=xgb_classifier,   
                min_features_to_select=min_num_feats_RFECV,
                scoring=scoring_RFECV,                           
                n_jobs=-1,                                  # parallelize cross-validation if possible
                cv=GroupKFold(
                    n_splits=num_folds_RFECV,
                    )
            ),
        )

    return model_pipeline
