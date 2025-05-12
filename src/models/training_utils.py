'''
Various tools for training and cross-validating models.
'''
from sklearn.pipeline import Pipeline

def cross_validation_wrapper(
    model_pipeline: Pipeline, 
    param_grid: dict,
    method: str = 'random',
    n_iter: int = 10,
    random_state: int = 591
):
    '''
    A cross-validation wrapper for hyperparameter tuning GBM and RF model pipelines.
    
    Parameters
    ----------
    
    model_pipeline: sklearn.pipeline.Pipeline
        A model pipeline.
        
    param_grid: dict
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or alternatively scipy.stats
        distributions to sample from.
        
    method: one of {'random','grid'}, default 'random'.
        * random: Perform randomized searching (`sklearn.model_selection.RandomizedSearchCV`) across parameter grid
        * grid: Perform exhaustive grid search (`sklearn.model_selection.GridSearchCV`)
          across all combinations of hyperparameters.
        
    .. Note:: if 'grid' is specified, param_grid must hold distinct lists of values rather than distributions.
    
    n_iter: int, default = 10
        Number of parameter configurations to test if using randomized search.
      
    random_state: int, default = 591
        Random seed for sampling parameter grid and cross-validation.  
    
     Returns
    -------
    dict
        A dictionary containing the results of the hyperparameter tuning:
        - 'best_model': the fitted sklearn pipeline with the best found parameters
        - 'best_score': the highest cross-validated score achieved
        - 'best_params': the parameter configuration corresponding to the best score
    '''
    
    
    
    