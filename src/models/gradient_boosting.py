'''
A pipeline, tools, and wrappers for feature selection, training, cross-validating, 
and predicting with Gradient Boosting models (XGBoost)
for usage on the CFS Remote Sensing Data.
'''
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE,RFECV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

def build_gbm_pipeline(
    feat_select: str = None,
    drop_features: list = [],
    num_feats_RFE: int = 4,
    min_num_feats_RFECV: int = 4,
    num_folds: int = 5,
    **kwargs
):
    '''
    A wrapper for building a Gradient Boosting model pipeline.
    
    Recursive Feature Elimination can be implemented to extract useful features.
    Handles column dropping, etc.
    
    Parameters
    ----------
    feat_select: {None, 'RFE', 'RFECV'}, default None
        Type of feature selection to be performed.
        
        * None: No feature engineering is performed; all features included in the model.
        * RFE: Recursive Feature Elimination is implemented 
          using an XGBoost model to measure feature importance.
          see sklearn.feature_selection.RFE documentaton for details.
          
        * RFECV: Similar to RFE with built-in cross validation using F1 Score.
          see sklearn.feature_selection.RFECV documentaton for details.
    
        .. Note::
            by default, half of the input features are included in the model if feature selection
            is specified. Control this using the `n_features_to_select` parameter. 
            
    drop_features: list, default = []
        Additional features to drop from model.
        Can be used following feature selection to simplify modelling.
    
    num_feats_RFE: int, default 4
        Number of features to keep if using the RFE algorithm.

    min_num_feats_RFECV: int, default 4
        Minimum number of features selected if using the RFECV algorithm.
        
    num_folds: int, default 5
        Number of cross-validation folds if using RFECV algorithm
        
    **kwargs: 
        Additonal arguments and hyperparameters passsed to XGBoost model.
        See xgboost.XGBClassifier documentaton for full hyperparameter specifications.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        A Scikit-Learn model pipeline.
    '''
    # columns to preprocess
    drop_cols = ['ID', 'PixelID'] + drop_features   # ID cols ignored, option to drop others
    categorical_cols = ['Type']                     # One-Hot Encode type columns  
    
    preprocessor = make_column_transformer(
            ('drop',drop_cols),
            (OneHotEncoder(),categorical_cols),
            remainder='passthrough'
        )
    
    if feat_select not in (None, 'RFE', 'RFECV'):
        raise ValueError(
            'feat_select must be one of: {None, \'RFE\', \'RFECV\'}'
            )
    
    # no feature selection
    if feat_select == None:
        model_pipeline = make_pipeline(
            preprocessor,
            XGBClassifier(**kwargs)
        )  
        
    # feature selection with RFE 
    elif feat_select == 'RFE':
        model_pipeline = make_pipeline(
            preprocessor,
            RFE(
                estimator = XGBClassifier(**kwargs),
                n_features_to_select=num_feats_RFE
            ),
            XGBClassifier(**kwargs)
        )
      
    # Feature selection with RFECV  
    else:
        model_pipeline = make_pipeline(
            preprocessor,
            RFECV(
                estimator = XGBClassifier(**kwargs),   
                min_features_to_select=min_num_feats_RFECV,
                scoring='f1',                                    # use f1 score to handle imbalance
                n_jobs=-1,                                       # parallelize cross-validation if possible
                cv=num_folds
            ),
            XGBClassifier(**kwargs)
        )

    return model_pipeline
