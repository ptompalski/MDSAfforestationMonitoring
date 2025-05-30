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


    predictions = torch.empty(0)
    for batch in test_dataloader:
        pred = model(
            batch['sequence'].to(device, non_blocking=True),
            batch['sequence_length'],
            batch['site_features'].to(device, non_blocking=True)
        )
        predictions = torch.cat((predictions, pred))
        
    pred_df = target_to_bin(test_set.lookup, threshold).rename(
        columns={'target': 'y_true'})
    pred_df['target'] = predictions.detach().numpy()
    pred_df = target_to_bin(pred_df, threshold).rename(
        columns={'target': 'y_pred'})
    
    return pred_df
    
def rnn_get_metrics(pred_df):
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

def main(trained_model_path,
         lookup_dir,
         seq_dir,
         threshold=0.7,
         batch_size=64,
         num_workers=0,
         site_cols='',
         seq_cols=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process CLI input for site_cols and seq_cols
    site_feats = ['Density', 'Type_Conifer', 'Type_Decidous', 'Age']
    seq_feats = ['NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR',
                 'TCB', 'TCG', 'TCW', 'log_dt', 'neg_cos_DOY']
    site_cols = site_feats if site_cols == '' else site_cols.split(',')
    seq_cols =  seq_feats if seq_cols == '' else seq_cols.split(',')
    
    
    # Load trained model
    checkpoint = torch.load(trained_model_path)
    config = checkpoint["config"]
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
    
    os.makedirs('results', exist_ok=True)
    output_name = f'eval_{config['rnn_type']}_{threshold*100}_{'' if config['concat_features'] else 'no_'}site_feats.pkl'
    output_path = os.path.join('results', output_name)
    with open(output_path, 'wb') as f:
        pickle.dump(rnn_test_result, f)
    print(f'Evaluation Completed. Results saved to {output_path}')

if __name__ == "__main__":
    main()
