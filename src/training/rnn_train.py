import rnn_dataset
import torch
import os
import click
import pandas as pd
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.rnn import RNNSurvivalPredictor

def train(model, train_dataloader, valid_dataloader, train_set, valid_set, device, optimizer, criterion, epochs=10, patience=5):

    valid_losses = []

    print(f'Training Model on {epochs} epochs on {device}.')
    model.to(device, non_blocking=True)
    for epoch in range(epochs):
        train_set.reshuffle()
        valid_set.reshuffle()
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            predictions = model(
                batch['sequence'].to(device, non_blocking=True),
                batch['sequence_length'],
                batch['site_features'].to(device, non_blocking=True)
                )
            train_loss = criterion(
                predictions.to(device, non_blocking=True),
                batch['target'].to(device, non_blocking=True)
            )
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                predictions = model(
                    batch['sequence'].to(device, non_blocking=True),
                    batch['sequence_length'],
                    batch['site_features'].to(device, non_blocking=True)
                )
                valid_loss = criterion(
                    predictions,
                    batch['target'].to(device, non_blocking=True)
                )
                total_valid_loss += valid_loss.item()
            avg_valid_loss = total_valid_loss/len(valid_dataloader)
            valid_losses.append(avg_valid_loss)

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Valid Loss = {avg_valid_loss:.4f}")

        # Early stopping check
        if epoch > 0 and avg_valid_loss > valid_losses[-2] * (1 + 1e-5):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model


@click.command()
@click.option('--model_path', type=click.Path(exists=True), required=True, help='Path to model .pth file.')
@click.option('--output_path', type=click.Path(), required=True, help='Path to save the trained model.')
@click.option('--lookup_dir', type=click.Path(exists=True), required=True, help='Directory to lookup file.')
@click.option('--data_dir', type=click.Path(exists=True), required=True, help='Directory to sequence data files.')
@click.option('--lr', type=float, default=0.01, help='Learning Rate of Adam optimizer.')
@click.option('--batch_size', type=int, default=32, help='Batch size for model.')
@click.option('--epochs', type=int, default=10, help='Number of epochs to train the model on.')
@click.option('--patience', type=int, default=5, help='Early stopping patience.')
@click.option('--num_workers', type=int, default=0, help='Number of workers for dataloader.')
@click.option('--pin_memory', type=bool, default=False, help='Whether to pin_memory before returning.')
@click.option('--site_cols', type=str, default='', help='Site features to use in model.')
@click.option('--seq_cols', type=str, default='', help='Sequence features to use in model.')
def main(model_path,
         output_path,
         lookup_dir,
         data_dir,
         lr=0.01,
         batch_size=64,
         epochs=10,
         patience=5,
         num_workers=0,
         pin_memory=False,
         site_cols='',
         seq_cols=''):
    
    TRAIN_LOOKUP_PATH = os.path.join(lookup_dir, 'train_lookup.parquet')
    VALID_LOOKUP_PATH = os.path.join(lookup_dir, 'valid_lookup.parquet')

    if site_cols == '':
        site_cols = ['Density', 'Type_Conifer', 'Type_Decidous', 'Age']
    else:
        site_cols = site_cols.split(',')
    
    if seq_cols == '':
        seq_cols = ['NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR',
                    'TCB', 'TCG', 'TCW', 'log_dt', 'neg_cos_DOY']
    else:
        seq_cols = seq_cols.split(',')
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

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
    valid_set, valid_dataloader = rnn_dataset.dataloader_wrapper(
        lookup_dir=VALID_LOOKUP_PATH,
        seq_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        site_cols=site_cols,
        seq_cols=seq_cols
    )

    model = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        patience=patience,
        train_set=train_set,
        valid_set=valid_set,
        device=device
    )


    torch.save(model, output_path)
    print(f'Training Complete, model saved to {output_path}.')

if __name__ == "__main__":
    main()
