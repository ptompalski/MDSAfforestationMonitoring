'''
A pipeline, tools, and wrappers for feature selection, training, cross-validating, 
and predicting with Gradient Boosting models (XGBoost)
for usage on the CFS Remote Sensing Data.
'''

def build_xgb_pipeline(
    feat_select: str = None,
    **kwargs
):
    '''
    A wrapper for building a Gradient Boosting model pipeline.
    
    Parameters
    ----------
    feat_select: {None, 'RFE', 'permutation'}, default None
        Type of feature selection to be performed.
        
        * None: No feature engineering is performed; all features included in the model.
        * RFE: Recursive Feature Elimination is implemented 
          using sklearn.feature_selection.RFE.
    '''
   