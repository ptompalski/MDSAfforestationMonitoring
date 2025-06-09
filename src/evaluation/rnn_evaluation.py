import os
import click
import pandas as pd
import sys
import torch
import pickle
from sklearn.metrics import (
    f1_score, fbeta_score, accuracy_score,
    precision_score, recall_score)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.pivot_data import target_to_bin
from evaluation.error_metrics import get_conf_matrix
from training.rnn_dataset import dataloader_wrapper
from models.rnn import RNNSurvivalPredictor


def rnn_get_prediction(model, test_dataloader, test_set, threshold, device):
    """
    Use the trained model to get survival rate predictions on test data. 
    Map the actual survival rates and predicted survival rates into 
    binary class Low/High (0/1) survival rate based on the given threshold.
    
    Parameters
    ----------
        model : torch.nn.Module
            The trained rnn model.
        test_dataloader : torch.utils.data.DataLoader
            DataLoader for test data.
        test_set : torch.utils.data.Dataset
            Full test dataset.
        threshold : float
            Survival rate classification threshold. Must be a value between 0 to 1.
        device : torch.device
            Device on which to run the model (CPU or CUDA).

    Returns
    ----------
    pd.DataFrame
        Dataframe containing:
        * Site Feature Columns: `ID`, `PixelID`, `SrvvR_Date`, Age, `Density`, `Type_Conifer`, `Type_Decidous`
        * Target Columns: `y_true` (True class labels), `y_pred` (Predicted class labels)
    """
    predictions = torch.empty(0)
    for batch in test_dataloader:
        pred = model(
            batch['sequence'].to(device, non_blocking=True),
            batch['sequence_length'],
            batch['site_features'].to(device, non_blocking=True)
        )
        predictions = torch.cat((predictions, pred))
    raw_y_true = test_set.lookup['target']
    raw_y_pred = predictions.detach().numpy()
    pred_df = target_to_bin(test_set.lookup, threshold).rename(
        columns={'target': 'y_true'})
    pred_df['target'] = raw_y_pred
    pred_df = target_to_bin(pred_df, threshold).rename(
        columns={'target': 'y_pred'})
    pred_df['raw_y_true'] = raw_y_true
    pred_df['raw_y_pred'] = raw_y_pred
    return pred_df
    
def rnn_get_metrics(pred_df):
    """
    This function computes various evaluation metrics for binary classification
    including F1 Score, F2 Score, Precision, Recall, Accuracy, Confusion Matrix and Class Proportions to assess imbalance.
    Note that class 0 (Low survival rate) is treated as the positive class.

    Parameters
    ----------
    pred_df : pd.DataFrame
        A DataFrame that must contain the following columns:
        - 'y_true' : True class labels (0 or 1)
        - 'y_pred' : Predicted class labels (0 or 1).
    
    Returns
    -------
    Tuple[dict, pd.DataFrame]
        metrics_dict : dict
            A dictionary containing:
            - 'F1 Score': F1 score
            - 'F2 Score': F2 score (β=2)
            - 'Precision': Precision score
            - 'Recall': Recall score
            - 'Accuracy': Accuracy score
            - '% Low Risk': Percentage of samples predicted as class 0 (Low survival rate)
            - '% High Risk': Percentage of samples predicted as class 1 (High survival rate)
        conf_matrix : pd.DataFrame
            A 2x2 labeled confusion matrix showing predicted vs. actual class counts.
    """
    y_true = pred_df['y_true']
    y_pred = pred_df['y_pred']
    
    # F1 and F2 score
    f1 = round(f1_score(y_true, y_pred, pos_label=0), 3)
    f2 = round(fbeta_score(y_true, y_pred, pos_label=0, beta=2), 3)

    # accuracy, precision, recall
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(
        y_true, y_pred, pos_label=0, zero_division=0), 3)
    recall = round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 3)

    # class proportions
    pct_low = float(round(sum(y_true == 0)/len(y_true), 3)*100)
    pct_high = float(round(sum(y_true == 1)/len(y_true), 3)*100)
    
    # Confusion matrix
    conf_matrix = get_conf_matrix(y_pred, y_true)
    
    metrics_dict = pd.Series({
        'F1 Score': f1,
        'F2 Score': f2,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        '% Low Risk': pct_low,
        '% High Risk': pct_high
    })

    return metrics_dict, conf_matrix

def rnn_get_metrics_by_age(pred_df):
    """
    Groups the test data by 'Age' (years since planting) and computes evaluation metrics
    for each age group.

    Parameters
    ----------
    pred_df : pd.DataFrame
        A DataFrame that must contain the following columns:
        - 'y_true' : True class labels (0 or 1)
        - 'y_pred' : Predicted class labels (0 or 1).
        - 'Age' : Number of years since planting (int).
    
    Returns
    -------
    Tuple[pd.DataFrame, dict[int, pd.DataFrame]]
        metrics_age : 
            DataFrame containing evaluation metrics for each age group (1–7), with columns:
            - 'Age': Number of years since planting
            - 'F1 Score': F1 score
            - 'F2 Score': F2 score (β=2)
            - 'Precision': Precision score
            - 'Recall': Recall score
            - 'Accuracy': Accuracy score
            - '% Low Survival Rate': Percentage of samples predicted as class 0 (Low survival rate)
            - '% High Survival Rate': Percentage of samples predicted as class 1 (High survival rate)
            - 'Number of Records': Total number of samples for the age group.
        conf_matrix : dict[int, pd.DataFrame]
            Dictionary mapping each age group (1-7) to its corresponding 2x2 labeled confusion matrix (Predicted vs. True class counts).
    
   Notes
    -----
    - Only results for age groups with available data are included in the output.
    """
    cols = ['F1 Score', 'F2 Score', 'Precision', 'Recall', 'Accuracy', '% Low Risk', '% High Risk']
    metrics_age = pd.DataFrame(index=range(1, 8), columns=cols)
    num_rec = []
    conf_matrix_age = {}
    for i in range(1, 8):
        try:
            age_df = pred_df[pred_df['Age'] == i]
            metrics_age.iloc[i-1], conf_matrix = rnn_get_metrics(age_df)
            num_rec.append(age_df.shape[0])
            conf_matrix_age[i] = conf_matrix
        except:
            pass
    metrics_age['Number of Records'] = num_rec
    return metrics_age.dropna().reset_index(names=['Age']), conf_matrix_age


@click.command()
@click.option('--trained_model_path', type=click.Path(exists=True), required=True, help='Path to trained model.')
@click.option('--eval_ouput_path', type=click.Path(exists=False), required=True, help='Path to save the evaluation results.')
@click.option('--lookup_dir', type=click.Path(exists=True), required=True, help='Directory to test lookup file.')
@click.option('--seq_dir', type=click.Path(exists=True), required=True, help='Directory to sequence data files.')
@click.option('--threshold', type=float, default=0.7, help='Survival rate threshold for target classification.')
@click.option('--batch_size', type=int, default=64, help='Batch size for test dataloader.')
@click.option('--num_workers', type=int, default=0, help='Number of workers for test dataloader.')
def main(trained_model_path,
         eval_output_path,
         lookup_dir,
         seq_dir,
         threshold=0.7,
         batch_size=64,
         num_workers=0):
    """
    Command Line Interface for evaluating trained rnn model on test data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    checkpoint = torch.load(trained_model_path)
    config = checkpoint["config"]
    site_cols = checkpoint['site_cols']
    seq_cols = checkpoint['seq_cols']
    model = RNNSurvivalPredictor(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device, non_blocking=True)
    model.eval()    
    
    # Load test data
    TEST_LOOKUP_PATH = os.path.join(lookup_dir, 'test_lookup.parquet')
    test_set, test_dataloader = dataloader_wrapper(
        lookup_dir=TEST_LOOKUP_PATH,
        seq_dir=seq_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        site_cols=site_cols,
        seq_cols=seq_cols
    )
    
    print(f'Getting predictions on test data ...')
    pred_df = rnn_get_prediction(
        model,
        test_dataloader,
        test_set,
        threshold,
        device
    )
    
    print(f'Evaluating overall model performance ...')
    metrics_overall, conf_matrix_overall = rnn_get_metrics(pred_df)
    
    print(f'Evaluating model performance by age group ...')
    metrics_age, conf_matrix_age = rnn_get_metrics_by_age(pred_df)
    
    rnn_test_result = {
        'error_metrics_overall': metrics_overall,
        'error_metrics_age' : metrics_age,
        'conf_matrix_overall' : conf_matrix_overall,
        'conf_matrix_age' : conf_matrix_age
    }
    
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
    with open(eval_output_path, 'wb') as f:
        pickle.dump(rnn_test_result, f)
    print(f'Evaluation Completed. Results saved to {eval_output_path}')

if __name__ == "__main__":
    main()
