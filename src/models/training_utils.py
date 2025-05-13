'''
Tools for training and cross-validating models.
'''
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, GroupKFold

def cross_validation_wrapper(
    model_pipeline: Pipeline, 
    df: pd.DataFrame,
    param_grid: dict,
    method: str = 'random',
    num_iter: int = 10,
    num_folds: int = 5,
    scoring: str = 'f1',
    return_results: bool = False,
    random_state: int = 591
):
    '''
    A cross-validation wrapper for hyperparameter tuning GBM and RF model pipelines.
    
    Parameters
    ----------
    
    model_pipeline: sklearn.pipeline.Pipeline
        A model pipeline.
        
    df: pd.DataFrame
        The cleaned remote sensing training data.
        Target survival rate should be converted to binary classes at this stage.xs
    
    param_grid: dict
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or alternatively scipy.stats
        distributions to sample from.
        
    method: one of {'random','grid'}, default='random'.
        * random: Perform randomized searching (`sklearn.model_selection.RandomizedSearchCV`) across parameter grid
        * grid: Perform exhaustive grid search (`sklearn.model_selection.GridSearchCV`)
          across all combinations of hyperparameters.
        
    .. Note:: if 'grid' is specified, param_grid must hold distinct lists of values rather than distributions.
    
    n_iter: int, default=10
        Number of parameter configurations to test if using randomized search.
      
    random_state: int, default=591
        Random seed for sampling parameter grid and cross-validation.  
    
    num_folds: int, default=5
        Number of folds used in cross-validation.
        Folds are split using `sklearn.model_selection.GroupKFold` to ensure sites
        do not overlap between folds.
    
    scoring: int, default='f1'
        Scoring metric used to rank hyperparameters during CV. 
        Must be a valid scoring string recognized by scikit-learn.
        See: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        
    return_results: bool, default=False
        if True, also returns scoring for every tested hyperparameter configuration
    
    random_state : int, default=591
        Random state seed for reproducibility.
    
    Returns
    -------
    dict
        A dictionary containing the results of the hyperparameter tuning:
        - 'best_model': the fitted sklearn pipeline with the best found parameters
        - 'best_score': the highest cross-validated score achieved
        - 'best_params': the parameter configuration corresponding to the best score
    '''
    # check for correct method specification
    if method not in ('random','grid'):
        raise ValueError(
            'method must be one of: {\'random\', \'grid\'}'
            )
    
    # get features and target
    X = df.drop(columns='target'); y = df['target']
    site_ids = df['ID']
    
    # cross validate by group k-fold to ensure site IDs do not overlap between each group
    group_kfold = GroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_state
        )
    
    if method == 'random':
        cross_validator = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=param_grid,
            n_iter=num_iter,
            scoring = scoring,
            n_jobs=-1,
            refit=True,
            cv = group_kfold
        )
    else:
        cross_validator = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            scoring = scoring,
            n_jobs=-1,
            refit=True,
            cv = group_kfold
        )
    
    cross_validator.fit(X,y,groups=site_ids)
    
    output_dict = {
        'best_model': cross_validator.best_estimator_,
        'best_score': cross_validator.best_score_,
        'best_params': cross_validator.best_params_,
    }
    
    # clean up results and store in dataframe, if return_results is True
    if return_results:
        results_df = pd.DataFrame(cross_validator.cv_results_)
        results_df = (
            results_df
            .drop(columns=results_df.filter(regex=r'^split\d+_test_score$').columns)
            .drop(columns=['params','rank_test_score'])
            .sort_values(axis=0,by='mean_test_score',ascending=False)
            .round(3)
            .reset_index(drop=True)
        )
        # clean up column names
        results_df.columns = results_df.columns.str.replace(r'^.*__', '', regex=True)
        
        output_dict['results'] = results_df
       
    return output_dict
    