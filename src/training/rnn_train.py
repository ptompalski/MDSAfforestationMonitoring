import rnn_dataset
import torch
import os
import click
import pandas as pd
import joblib
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.rnn import RNNSurvivalPredictor

def train(model, train_dataloader, test_dataloader, train_set, test_set, optimizer, criterion, epoches=10, patience=5, device='cpu'):
    train_losses = []
    test_losses = []
    for epoch in range(epoches):
        train_set.reshuffle()
        test_set.reshuffle()
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            predictions = model(
                batch['sequence'], batch['sequence_length'], batch['site_features'])
            train_loss = criterion(predictions, batch['target'])
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                predictions = model(
                    batch['sequence'], batch['sequence_length'], batch['site_features'])
                test_loss = criterion(predictions, batch['target'])
                total_test_loss += test_loss.item()
            avg_test_loss = total_test_loss/len(test_dataloader)
            test_losses.append(avg_test_loss)

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

        # Early stopping check
        if epoch > 0 and avg_test_loss > test_losses[-2] * (1 + 1e-5):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0

        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model, train_losses, test_losses


@click.command()
@click.option('--model_path', type=click.Path(exists=True), required=True, help='Path to model joblib file.')
@click.option('--output_dir', type=click.Path(exists=True), required=True, help='Directory to save trained model.')
@click.option('--lookup_dir', type=click.Path(exists=True), required=True, help='Directory to lookup file.')
@click.option('--data_dir', type=click.Path(exists=True), required=True, help='Directory to sequence data files.')
@click.option('--lr', type=float, default=0.01, help='Learning Rate of Adam optimizer.')
@click.option('--batch_size', type=int, default=32, help='Batch size for model.')
@click.option('--epoches', type=int, default=10, help='Number of epoches to train the model on.')
@click.option('--patience', type=int, default=5, help='Early stopping patience.')
@click.option('--num_workers', type=int, default=0, help='Number of workers for dataloader.')
@click.option('--pin_memory', type=bool, default=False, help='Whether to pin_memory before returning.')
@click.option('--site_cols', type=list, default=None, help='Site features to use in model.')
@click.option('--seq_cols', type=list, default=None, help='Sequence features to use in model.')
def main(model_path,
         output_dir,
         lookup_dir,
         data_dir,
         lr=0.01,
         batch_size=32,
         epoches=10,
         patience=5,
         num_workers=0,
         pin_memory=False,
         site_cols=[],
         seq_cols=[]):
    
    TRAIN_LOOKUP_PATH = os.path.join(lookup_dir, 'train_lookup.parquet')
    TEST_LOOKUP_PATH = os.path.join(lookup_dir, 'test_lookup.parquet')
    if site_cols == []:
        site_cols = ['Density', 'Type_Conifer', 'Type_Decidous', 'Age']
    else:
        site_cols = site_cols.split(',') 
    if seq_cols == []:
        seq_cols = ['NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR',
                'TCB', 'TCG', 'TCW', 'log_dt', 'neg_cos_DOY'] 

    

    model = torch.load(model_path, weights_only=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    train_set, train_dataloader = rnn_dataset.dataloader_wrapper(
        lookup_dir=TRAIN_LOOKUP_PATH,
        seq_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        site_cols=site_cols,
        seq_cols=seq_cols
    )
    test_set, test_dataloader = rnn_dataset.dataloader_wrapper(
        lookup_dir=TEST_LOOKUP_PATH,
        seq_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        site_cols=site_cols,
        seq_cols=seq_cols
    )

    print(f'Training Model on {epoches} epoches.')
    model, train_losses, test_losses = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epoches=epoches,
        patience=patience,
        train_set=train_set,
        test_set=test_set
    )

    results = pd.DataFrame(
        {'Train Losses': train_losses,
         'Test Losses': test_losses}
    )
    # results.to_parquet(os.path.join('output_dir', 'rNN_result.parquet'))
    
    joblib.dump(model, os.path.join(output_dir, 'trained_rnn.joblib'))
    print(f'Training Complete, model saved to {output_dir}.')



if __name__ == "__main__":
    main()
