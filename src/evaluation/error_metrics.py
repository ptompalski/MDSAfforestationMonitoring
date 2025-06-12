'''
Utility functions for getting error metrics, including PR and ROC curves as well as 
F1 scores, accuracy, and AUC.
'''
from sklearn.model_selection import cross_val_predict,GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve, confusion_matrix,
    f1_score, fbeta_score, accuracy_score, 
    precision_score, recall_score)
from sklearn.pipeline import Pipeline
import pandas as pd
import click
import numpy as np
from pathlib import Path
import joblib

def get_validation_preds(
    model: Pipeline,
    df_train: pd.DataFrame,
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
    dict
        A dictionary containing:
        - y_pred: Predicted class labels.
        - y_prob: Predicted probabilities for the class labelled as 1 (high survival rate).
        - y_true: True binary labels (0 for 'Low Survival Rate' or 1 for 'High Survival Rate').
    '''
    # get features and target
    df_train = df_train.dropna()
    X = df_train.drop(columns='target'); y_true = df_train['target']
    site_ids = df_train['ID']
    
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
        'y_prob':y_prob[:, 0], # Probability of class 0 ("Low survival rate" = positive class)
        'y_true':y_true.values
        }
    
def get_test_errors(
    model: Pipeline,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    random_state: int = 591  
):
    '''
    Train the given model on the given training data and
    output error metrics on the test data.
    
    This allows for more precise error analysis, eg. for comparing test error across site years (`Age`) 
    
    Parameters
    ----------
    model: sklearn.pipeline.Pipeline
        The model pipeline. The model should have been and tuned for optimal hyperparameters
        at this stage.
        
    df_train: pd.DataFrame
        The input training data.
        
    df_test: pd.DataFrame
        The input test data.
        
    Returns
    -------
    dict
        A dictionary containing:
        - y_pred: Predicted class labels on test data.
        - y_prob: Predicted probabilities for the class labelled as 1 (high survival rate).
        - y_true: True binary labels (0 for 'Low Survival Rate' or 1 for 'High Survival Rate') of test data.
    '''
    # split training and testinng data 
    X_train = df_train.drop(columns='target'); y_train = df_train['target']
    X_test = df_test.drop(columns='target'); y_test = df_test['target']
    
    # fit model on training data
    model.fit(X_train,y_train)
    
    # get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # get error metrics
    test_errors = get_error_metrics(y_pred,y_prob[:, 0],y_test)
    
    return test_errors

def get_valid_roc_curve(
    y_prob: np.array, 
    y_true: np.array
):  
    '''
    Return a dataframe containing points along the ROC curve.

    This function computes the Receiver Operating Characteristic (ROC) curve for binary classification. 
    Note the 'Low survival rate' class (label 0) is treated as the positive class.

    Parameters
    ----------
    y_prob : np.array
        Predicted probabilities for the class labelled as 0 (e.g., low survival).

    y_true : np.array
        True binary labels (0 or 1).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - 'False Positive Rate': Array of FPR values
        - 'True Positive Rate': Array of TPR values
        - 'Thresholds': Decision thresholds used to compute FPR/TPR
    '''
    # need to flip labels, as we should consider 'Low survival rate' (with labels 0) to be positive cases
    fpr,tpr, threshold = roc_curve(y_true, y_prob,pos_label=0)

    return pd.DataFrame({
        'False Positive Rate':fpr,
        'True Positive Rate': tpr,
        'Thresholds': threshold
    })

def get_valid_pr_curve(
    y_prob: np.array,
    y_true: np.array, 
):
    '''
    Return a dataframe containing points along the Precision-Recall curve.

    This function computes the Precision-Recall (PR) curve for binary classification.
    Note the 'Low survival rate' class (label 0) is treated as the positive class.

    Parameters
    ----------
    y_prob : np.array
        Predicted probabilities for the class labelled as 0 (e.g., low survival).
        
    y_true : np.array
        True binary labels (0 or 1).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - 'Precision': Precision values
        - 'Recall': Recall values
        - 'Thresholds': Decision thresholds used to compute Precision/Recall
    '''
    # need to flip labels, as we should consider 'Low survival rate' (with labels 0) to be positive cases
    precision,recall,threshold = precision_recall_curve(y_true, y_prob,pos_label=0)

    return pd.DataFrame({
        'Precision':precision[:-1],
        'Recall': recall[:-1],
        'Thresholds': threshold
    })

def get_conf_matrix(
    y_pred: np.array,
    y_true: np.array, 
):
    '''
    Compute the confusion matrix for predicted and true labels.

    This function returns the confusion matrix treating class 0 ('Low survival rate') 
    as the positive class. The matrix is returned as a pandas DataFrame for readability.

    Parameters
    ----------
    y_true : np.array
        True binary labels (0 for 'low survival rate' or 1 for 'high survival rate').

    y_pred : np.array
        Predicted binary class labels.

    Returns
    -------
    pd.DataFrame
        A 2x2 labeled confusion matrix showing predicted vs. actual class counts.
    '''
    conf_mat = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=[0, 1]))
    conf_mat.index = ['True Low','True High']
    conf_mat.columns = ['Predicted Low','Predicted High']
    return conf_mat
    
def get_error_metrics(y_pred: np.array, y_prob: np.array, y_true: np.array):
    '''
    Compute and return a dictionary of classification error metrics.

    This function computes various evaluation metrics for binary classification
    including F1 Score, F2 Score, Precision, Recall, Accuracy, ROC AUC,
    Average Precision (AP), and class proportions to assess imbalance.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels: 0 for 'Low Survival Rate' or 1 for 'High Survival Rate'
        Note that 'Low Survival Rate' is considered the positive class.
    
    y_pred : np.ndarray
        Predicted class labels.
    
    y_prob : np.ndarray
        Predicted probabilities for the positive class (pos_label).
    
    Returns
    -------
    dict
        A dictionary containing:
        - 'F1 Score': F1 score
        - 'F2 Score': F2 score (β=2)
        - 'Precision': Precision score
        - 'Recall': Recall score
        - 'Accuracy': Accuracy score
        - 'AUC': ROC AUC score
        - 'AP': Average Precision score
        - '% Low Rate': Percentage of samples predicted as class 0
        - '% High Rate': Percentage of samples predicted as class 1
    '''
    # ROC and AP scores
    # Need to flip label, as these functions expect 1 to be the positive class
    # and pos_label cannot be set. 
    roc_auc = float(round(roc_auc_score(1 - y_true, y_prob), 3))
    ap_score = float(round(average_precision_score(1 - y_true, y_prob), 3))
    
    # F1 and F2 score
    f1 = round(f1_score(y_true,y_pred,pos_label=0),3)
    f2 = round(fbeta_score(y_true,y_pred,pos_label=0,beta=2),3)
    
    # accuracy, precision, recall
    accuracy = round(accuracy_score(y_true,y_pred),3)
    precision = round(precision_score(y_true,y_pred,pos_label=0),3)
    recall = round(recall_score(y_true,y_pred,pos_label=0),3)
    
    # class proportions
    pct_low = float(round(sum(y_true == 0)/len(y_true),3)*100)   
    pct_high = float(round(sum(y_true == 1)/len(y_true),3)*100)   
    
    return {
        'F1 Score':f1,
        'F2 Score':f2,
        'Precision':precision,
        'Recall': recall,
        'Accuracy':accuracy,
        'AUC':roc_auc,
        'AP': ap_score,
        '% Low Rate': pct_low,
        '% High Rate': pct_high
    }

@click.command()
@click.option('--tuned_model_path', type=click.Path(exists=True), required=True, help='Path to tuned model.')
@click.option('--training_data_path', required=True, help='Path to training parquet file')
@click.option('--output_dir', type=click.Path(file_okay=False), required=True, help='Directory to save the evaluation results.')
def main(tuned_model_path,training_data_path,output_dir):
    '''
    CLI for reporting error metrics of classical ML models (RF, GBM, LR)
    '''
    # load in model, training data
    model = joblib.load(tuned_model_path)
    df_train = pd.read_parquet(training_data_path)
    model_name = Path(tuned_model_path).name.removeprefix('tuned_').removesuffix('.joblib')
    
    # make a path to store results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True,parents=True)
    
    # get predictions
    click.echo(f"Getting validation predictions for {model_name}...")
    valid_pred_dict = get_validation_preds(model,df_train)
    y_pred = valid_pred_dict['y_pred']
    y_prob = valid_pred_dict['y_prob']
    y_true = valid_pred_dict['y_true']
    
    # get error metrics
    click.echo("Getting scores...")
    scores = get_error_metrics(y_pred,y_prob,y_true)
    scores_fname = f"{model_name}_scores.joblib"
    joblib.dump(scores,output_dir/scores_fname)
    click.echo(f"Scores saved to {output_dir/scores_fname}")
    
    # get PR curve
    click.echo("Getting PR curve...")
    pr_curve = get_valid_pr_curve(y_prob,y_true)
    pr_curve_fname = f"{model_name}_pr_curve.csv"
    pr_curve.to_csv(output_dir/pr_curve_fname,index=False)
    click.echo(f"PR Curve saved to {output_dir/pr_curve_fname}")
    
    # get ROC curve
    click.echo("Getting ROC curve...")
    roc_curve = get_valid_roc_curve(y_prob,y_true)
    roc_curve_fname = f"{model_name}_roc_curve.csv"
    roc_curve.to_csv(output_dir/roc_curve_fname,index=False)
    click.echo(f"ROC Curve saved to {output_dir/roc_curve_fname}")
    
    # get Confusion Matrix
    click.echo("Getting Confusion Matrix...")
    conf_matrix = get_conf_matrix(y_pred,y_true)
    conf_matrix_fname = f"{model_name}_confusion_matrix.csv"
    conf_matrix.to_csv(output_dir/conf_matrix_fname)
    click.echo(f"Confusion matrix saved to {output_dir/conf_matrix_fname}")
    
if __name__ == '__main__':
    main()