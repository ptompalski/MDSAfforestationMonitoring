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

