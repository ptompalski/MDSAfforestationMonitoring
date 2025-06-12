'''
Tools for training and cross-validating models.
'''
import click
import joblib
import pandas as pd
import json
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer,f1_score
from scipy.stats import randint,loguniform,uniform
import numpy as np

def _get_rand_hparam_grid(model_pipeline: Pipeline):
    '''
    A simple helper function to return a suitable hyperparameter grid based on the given model pipeline,
    for a random search cross validation.
    
    Parameters
    ---------
    model_pipeline: sklearn.pipeline.Pipeline
        A model pipeline to be cross-validated. 
        Should be one of the Random Forest, Gradient Boosting, or Logistic Regression Models
        constructed via the `gradient_boosting.py`, `logistic_regression.py` or `random_forest.py` scripts.
        
    Returns
    -------
    dict:
        A dictionary containing a suitable hyperparameter gridf. For example, for an XGBoost model, returns:
            {
            'xgbclassifier__max_depth': randint(2,10),
            'xgbclassifier__learning_rate': uniform(0.01,0.3),
            'xgbclassifier__n_estimators': randint(500,2000),
            'xgbclassifier__reg_alpha': loguniform(1e-4,1e2),
            'xgbclassifier__reg_lambda': loguniform(1e-4,1e2)
            }
    '''
    if 'xgbclassifier' in model_pipeline.named_steps:
        return {
            'xgbclassifier__max_depth': randint(2,10),
            'xgbclassifier__learning_rate': uniform(0.01,0.3),
            'xgbclassifier__n_estimators': randint(500,2000),
            'xgbclassifier__reg_alpha': loguniform(1e-4,1e2),
            'xgbclassifier__reg_lambda': loguniform(1e-4,1e2)
        }
    elif 'randomforestclassifier' in model_pipeline.named_steps:
        return {
            "randomforestclassifier__criterion": ['gini', 'entropy', 'log_loss'],
            "randomforestclassifier__max_depth": randint(1,20),
            "randomforestclassifier__bootstrap": [True, False],
            "randomforestclassifier__class_weight": ['balanced', None],
            'randomforestclassifier__n_estimators': randint(100,1000),
            "randomforestclassifier__max_features": ['sqrt', 'log2', None],
            "randomforestclassifier__n_jobs": [-1],
        }
    elif 'logisticregression' in model_pipeline.named_steps:
        return {
            "logisticregression__C": loguniform(1e-4, 1e2),
            "logisticregression__penalty": ["l2",'l1','elasticnet',None],
            "logisticregression__class_weight": ['balanced', None],
            'logisticregression__max_iter': randint(5000,10000)
        }
    else:
        raise ValueError(
            'Incorrect model specification: Expecting xgbclassifier, randomforestclassifier, or logisticregression.'
        )

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
    
    scoring: str, default='f1'
        Scoring metric used to rank hyperparameters during CV. 
        Must be a valid scoring string recognized by scikit-learn.
        See: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        
    return_results: bool, default=False
        if True, also returns a dataframe with performance metrics for every tested hyperparameter configuration.
    
    random_state : int, default=591
        Random state seed for reproducibility.
    
    Returns
    -------
    dict
        A dictionary containing the results of the hyperparameter tuning:
        - 'best_model': the fitted sklearn pipeline with the best found parameters
        - 'best_score': the highest cross-validated score achieved
        - 'best_params': the parameter configuration corresponding to the best score
        - 'results': (optional) A dataframe with cross-validation results.
    '''
    # check for correct method specification
    if method not in ('random','grid'):
        raise ValueError(
            'method must be one of: {\'random\', \'grid\'}'
            )

    # get features and target
    df = df.dropna()
    X = df.drop(columns='target'); y = df['target']
    site_ids = df['ID']

    # cross validate by group k-fold to ensure site IDs do not overlap between each group
    group_kfold = GroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_state
        )
    
    # make sure default scorer takes 0 as positive label
    scoring = make_scorer(f1_score,pos_label=0) if scoring == 'f1' else scoring
    
    if method == 'random':
        cross_validator = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=param_grid,
            n_iter=num_iter,
            scoring = scoring,
            n_jobs=-1,
            refit=True,
            cv = group_kfold,
            random_state=random_state
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
    
    cross_validator.fit(X, y, groups=site_ids)
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

@click.command()
@click.option('--model_path', required=True,
              help='Directory to load pipeline model')
@click.option('--training_data', required=True, help='Directory to training parquet file')
@click.option('--tuning_method', type=click.Choice(['grid', 'random'], case_sensitive=False), required=True, help='Method for tuning the model. Options: grid, random')
@click.option('--param_grid', 
              help='''Parameter grid for tuning the model. 
              Should be a dictionary with parameter names as keys and lists of values as values.
              Default: use pre-defined parameter distributions.''',
              default='default'
              )
@click.option('--num_iter', type=int, default=10, help='Number of parameter configurations to test if using randomized search')
@click.option('--num_folds', type=int, default=5, help='Number of folds used in cross-validation')
@click.option('--scoring', type=str, default='f1', help='Scoring metric used to rank hyperparameters during CV')
@click.option('--random_state', type=int, default=591, help='Random seed for reproducibility')
@click.option('--return_results', type=bool, default=False, help='Whether to return cross-validation results')
@click.option('--output_dir', type=click.Path(file_okay=False), help='Directory to save the tuning results')
def main(model_path, training_data, tuning_method, param_grid,
         num_iter, num_folds, scoring, random_state, output_dir, 
         return_results):
    '''
    CLI for cross-validation of models.
    '''
    pipeline = joblib.load(model_path)
    df_train = pd.read_parquet(training_data)
    
    if tuning_method == 'random' and param_grid == 'default':
        param_grid = _get_rand_hparam_grid(pipeline)
    else:
        param_grid=json.loads(param_grid)
    
    print('Cross-validating...') 
    result = cross_validation_wrapper(
        model_pipeline=pipeline,
        df=df_train,
        param_grid=param_grid,
        method=tuning_method,
        num_iter=num_iter,
        num_folds=num_folds,
        scoring=scoring,
        random_state=random_state,
        return_results=return_results
    )
    model_name = f"tuned_{model_path.split('/')[-1]}"
    output_dir = os.path.join(output_dir, training_data.split('/')[-2])
    model_path = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(result['best_model'], model_path)
    print(f"Model saved to {model_path}")
    
    if return_results:
        os.makedirs(os.path.join(output_dir,'logs'),exist_ok=True)
        result_path = os.path.join(
            output_dir,'logs',
            f'{model_name.split('.')[-2]}_log.csv')
        result['results'].to_csv(result_path,index=False)
        print(f"tuning log saved to {result_path}")
        
if __name__ == '__main__':
    main()