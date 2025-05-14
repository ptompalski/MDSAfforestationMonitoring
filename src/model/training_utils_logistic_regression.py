"""
Tools for training and cross-validating **logistic regression** model pipelines.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, GroupKFold

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
    A cross-validation wrapper for hyper-parameter tuning of logistic-regression model pipelines.
    
    Parameters
    ----------
    model_pipeline: sklearn.pipeline.Pipeline
        A logistic regression pipeline.
        
    df: pd.DataFrame
        The cleaned remote sensing training data.
        Must include a binary `target` column and an `ID` column for grouping.
    
    param_grid: dict
        Dictionary with parameter names (`str`) as keys and lists of
        values or scipy.stats distributions to sample from.
        
    method: one of {'random','grid'}, default='random'
        - 'random': RandomizedSearchCV across the parameter grid.
        - 'grid': GridSearchCV across all parameter combinations.
    
    num_iter: int, default=10
        Number of sampled parameter configurations if using randomized search.
      
    num_folds: int, default=5
        Number of GroupKFold splits to use (groups defined by `ID`).
    
    scoring: str, default='f1'
        Scoring metric for hyperparameter ranking.
    
    return_results: bool, default=False
        If True, returns a DataFrame of CV results under key 'results'.
    
    random_state: int, default=591
        Seed for reproducibility.
    
    Returns
    -------
    dict
        - 'best_model': fitted Pipeline with top parameters
        - 'best_score': highest cross-validated score
        - 'best_params': corresponding parameter set
        - 'results' (optional): cleaned DataFrame of CV outcomes
    '''
    # validate method
    if method not in ('random', 'grid'):
        raise ValueError("method must be one of {'random', 'grid'}")

    # prepare data
    X = df.drop(columns='target')
    y = df['target']
    groups = df['ID']

    # define cross-validation
    group_kfold = GroupKFold(n_splits=num_folds)

    # set up searcher
    if method == 'random':
        searcher = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=param_grid,
            n_iter=num_iter,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            cv=group_kfold,
            random_state=random_state
        )
    else:
        searcher = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            cv=group_kfold
        )

    # execute search
    searcher.fit(X, y, groups=groups)

    # gather outputs
    output = {
        'best_model': searcher.best_estimator_,
        'best_score': searcher.best_score_,
        'best_params': searcher.best_params_
    }

    if return_results:
        results_df = pd.DataFrame(searcher.cv_results_)
        # drop individual split scores and rank
        drop_cols = [c for c in results_df if c.startswith('split') and c.endswith('_test_score')]
        results_df = results_df.drop(columns=drop_cols + ['params', 'rank_test_score'])
        results_df = results_df.sort_values('mean_test_score', ascending=False).reset_index(drop=True)
        # simplify column names
        results_df.columns = results_df.columns.str.replace(r'^.*__', '', regex=True)
        output['results'] = results_df

    return output