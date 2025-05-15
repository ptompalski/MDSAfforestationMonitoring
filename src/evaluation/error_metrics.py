'''
Utility functions for getting error metrics, including PR and ROC curves as well as 
F1 scores, accuracy, and AUC.
'''
from sklearn.model_selection import cross_val_predict,GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
import click
import numpy as np

def get_preds_and_truth(
    model: Pipeline,
    df: pd.DataFrame,
    num_folds: int = 5,
    random_state: int = 591
):
    '''
    Helper function that trains the given model using cross-validation.
    probabilistic and class predictions are then made on the hold-out fold.
    
    Parameters
    ----------
    
    model: sklearn.pipeline.Pipeline
        The model pipeline. The model should have been and tuned for optimal hyperparameters
        at this stage.
        
    df_train: pd.DataFrame
        The input training data.
        
    num_folds: int, default=5
        The number of folds to use during cross validation.
        
    random_state: int, default=591
        Random state seed for reproducibility.
        
    Returns
    -------
    tuple: (y_pred, y_pred_proba, y_true)
        A tuple of three one-dimensional arrays:
        - y_pred: directly predict binary class labels.
        - y_pred_proba: output probabilistic predictions for ROC and PR curves
        - y_true: The ground-truth class labels.
    '''
    
    # get features and target
    X = df.drop(columns='target'); y_true = df['target']
    site_ids = df['ID']
    
    # cross validate by group k-fold to ensure site IDs do not overlap between each group
    group_kfold = GroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_state
        )
    
    # Get class predictions
    y_pred = cross_val_predict(
        model, X, y_true, 
        cv=group_kfold, 
        method='predict',
        groups=site_ids)

    # Get predicted probabilities
    y_prob = cross_val_predict(
        model, X, y_true, 
        cv=group_kfold, 
        method='predict_proba',
        groups=site_ids)
    
    return {
        'y_pred':y_pred,
        'y_prob':y_prob[:, 1], # want binary output, probability of high survival rate
        'y_true':y_true.to_numpy()
        }
    
def get_class_imbalance()

def get_valid_roc_curve(y_prob: np.array, y_true: np.array):
    # need to flip labels, as we should consider 'Low survival rate' to be positive cases (1)
    return roc_curve(1 - y_true, 1 - y_prob)


def get_valid_pr_curve(y_prob: np.array, y_true: np.array):
    return precision_recall_curve(1 - y_true, 1 - y_prob)

def get_error_metrics(y_pred: np.array, y_prob: np.array, y_true: np.array):
    
    # AUC score (area under ROC)
    roc_auc = roc_auc_score(1 - y_true, 1 - y_prob)
    
    # AP score (area under PR curve)
    ap_score = average_precision_score(1 - y_true, 1 - y_prob)
    
    # F1 score
    f1 = f1_score(y_true,y_pred,pos_label=0)
    
    # accuracy
    accuracy = accuracy_score(y_true,y_pred)
    
    return {
        'F1 Score':f1,
        'Accuracy':accuracy,
        'AUC':roc_auc,
        'AP': ap_score,
    }


def get_conf_matrix(y_pred: np.array, y_true: np.array):
    return confusion_matrix(y_true,y_pred)