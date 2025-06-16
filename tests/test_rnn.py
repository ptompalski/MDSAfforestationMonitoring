import pytest
import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.rnn import RNNSurvivalPredictor
from torch import nn


def test_rnn_model_init_GRU():
    """
    Test the initialization of the RNNSurvivalPredictor model using GRU.
    """
    input_size = 10
    hidden_size = 20
    linear_size = 16
    site_features_size = 5
    rnn_type = "GRU"
    num_layers = 2
    dropout_rate = 0.3

    model = RNNSurvivalPredictor(input_size, hidden_size, linear_size, site_features_size, rnn_type, num_layers, dropout_rate)

    assert model.rnn_layers == num_layers
    assert model.rnn_hidden_size == hidden_size
    assert model.rnn_type == rnn_type
    assert isinstance(model.rnn, nn.GRU)


def test_rnn_model_init_LSTM():
    """
    Test the initialization of the RNNSurvivalPredictor model using LSTM.
    """
    input_size = 10
    hidden_size = 20
    site_features_size = 5
    linear_size = 16
    rnn_type = "LSTM"
    num_layers = 2
    dropout_rate = 0.3

    model = RNNSurvivalPredictor(input_size, hidden_size, linear_size, site_features_size, rnn_type, num_layers, dropout_rate)

    assert model.rnn_layers == num_layers
    assert model.rnn_hidden_size == hidden_size
    assert model.rnn_type == rnn_type
    assert isinstance(model.rnn, nn.LSTM)


def test_rnn_model_forward():
    """
    Test the forward pass of the RNNSurvivalPredictor model.
    """
    input_size = 10
    hidden_size = 20
    linear_size =16
    site_features_size = 5
    rnn_type = "GRU"
    num_layers = 2
    dropout_rate = 0.3

    model = RNNSurvivalPredictor(input_size, hidden_size, linear_size, site_features_size, rnn_type, num_layers, dropout_rate)
    batch_size = 4
    seq_length = 6
    sequence = torch.randn(batch_size, seq_length, input_size)
    sequence_length = torch.tensor([seq_length] * batch_size)
    site_features = torch.randn(batch_size, site_features_size)

    output = model(sequence, sequence_length, site_features)
    assert output.shape == (batch_size,)


def test_rnn_model_invalid_rnn_type():
    """
    Test the initialization of the RNNSurvivalPredictor model with an invalid RNN type.
    """
    input_size = 10
    hidden_size = 20
    linear_size = 16
    site_features_size = 5
    rnn_type = "INVALID_RNN_TYPE"
    num_layers = 2
    dropout_rate = 0.3

    with pytest.raises(ValueError):
        RNNSurvivalPredictor(input_size, hidden_size, linear_size, site_features_size, rnn_type, num_layers, dropout_rate)


def test_rnn_model_concat_features():
    """
    Test the forward pass of the RNNSurvivalPredictor model with concatenated features.
    """
    input_size = 10
    hidden_size = 20
    site_features_size = 5
    linear_size = 16
    rnn_type = "GRU"
    num_layers = 2
    dropout_rate = 0.3

    model = RNNSurvivalPredictor(input_size, hidden_size, linear_size, site_features_size, 
                                 rnn_type, num_layers, dropout_rate, concat_features=True)
    batch_size = 4
    seq_length = 6
    sequence = torch.randn(batch_size, seq_length, input_size)
    sequence_length = torch.tensor([seq_length] * batch_size)
    site_features = torch.randn(batch_size, site_features_size)

    output = model(sequence, sequence_length, site_features)
    assert output.shape == (batch_size,)


def test_rnn_model_no_concat_features():
    """
    Test the forward pass of the RNNSurvivalPredictor model without concatenated features.
    """
    input_size = 10
    hidden_size = 20
    linear_size = 16
    site_features_size = 5
    rnn_type = "GRU"
    num_layers = 2
    dropout_rate = 0.3

    model = RNNSurvivalPredictor(input_size, hidden_size, linear_size, site_features_size, rnn_type, num_layers, dropout_rate, concat_features=False)
    batch_size = 4
    seq_length = 6
    sequence = torch.randn(batch_size, seq_length, input_size)
    sequence_length = torch.tensor([seq_length] * batch_size)
    site_features = torch.randn(batch_size, site_features_size)

    output = model(sequence, sequence_length, site_features)
    assert output.shape == (batch_size,)
