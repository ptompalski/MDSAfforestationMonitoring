import torch
import click
import os
import joblib
from torch import nn
import torch.nn.utils.rnn as rnn_utils


class RNNSurvivalPredictor(nn.Module):
    """
    A recurrent neural network (RNN) model for survival prediction.
    This class implements a survival prediction model using either GRU or LSTM recurrent layers.
    It processes sequential data along with static site features to predict survival rates.
    Attributes:
        rnn_layers (int): Number of recurrent layers.
        rnn_hidden_size (int): Size of the hidden state in the RNN.
        rnn_type (str): Type of RNN to use ('GRU' or 'LSTM').
        rnn (nn.Module): The recurrent neural network layer (GRU or LSTM).
        activation (nn.ReLU): Activation function for the output.
        linear (nn.Linear): Linear layer for final prediction.
        dropout (nn.Dropout): Dropout layer for regularization.
    Args:
        input_size (int): The number of features in each time step of the input sequence.
        hidden_size (int): The size of the hidden state in the RNN.
        site_features_size (int): The number of static features per site.
        rnn_type (str, optional): The type of RNN to use ('GRU' or 'LSTM'). Defaults to "GRU".
        num_layers (int, optional): Number of recurrent layers. Defaults to 1.
        dropout_rate (float, optional): Dropout probability for regularization. Defaults to 0.2.
    """
    def __init__(self, input_size, hidden_size,
                 site_features_size, rnn_type="GRU",
                 num_layers=1, dropout_rate=0.2, concat_features=False):
        super(RNNSurvivalPredictor, self).__init__()

        self.rnn_layers = num_layers
        self.rnn_hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.concat_features = concat_features

        if rnn_type not in ['GRU', 'LSTM']:
            raise ValueError("rnn_type must be either 'GRU' or 'LSTM'")

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(site_features_size + hidden_size if self.concat_features else hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sequence, sequence_length, site_features):
        batch_size = sequence.size(0)
        h0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(sequence.device)

        packed_input = rnn_utils.pack_padded_sequence(
                        sequence,
                        sequence_length.cpu().long(),
                        batch_first=True,
                        enforce_sorted=False)

        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(sequence.device)
            packed_output, (hn, cn) = self.rnn(packed_input, (h0, c0))
        else:
            packed_output, hn = self.rnn(packed_input, h0)

        last_hidden_state = hn[-1]
        concatenated_features = torch.cat((last_hidden_state, site_features), dim=1) if self.concat_features else last_hidden_state
        input_dropped = self.dropout(concatenated_features)
        hidden_output = self.activation(input_dropped)
        hidden_dropped = self.dropout(hidden_output)
        output = self.linear(hidden_dropped)

        clamped_output = torch.clamp(output, 0, 100)
        return clamped_output.squeeze()

@click.command()
@click.option('--input_size', type=int, required=True, help='Number of features in each time step of the input sequence')
@click.option('--hidden_size', type=int, required=True, help='Size of the hidden state in the RNN')
@click.option('--site_features_size', type=int, required=True, help='Number of static features per site')
@click.option('--rnn_type', type=click.Choice(['GRU', 'LSTM']), default='GRU', help='Type of RNN to use')
@click.option('--num_layers', type=int, default=1, help='Number of recurrent layers')
@click.option('--dropout_rate', type=float, default=0.2, help='Dropout probability for regularization')
@click.option('--concat_features', type=bool, default=False, help='Concatenate site features with RNN output')
@click.option('--output_dir', type=click.Path(file_okay=False), required=True, help='Directory to save the model')
def main(input_size, hidden_size, site_features_size, rnn_type, num_layers, dropout_rate, concat_features, output_dir):
    model = RNNSurvivalPredictor(input_size, hidden_size, site_features_size, rnn_type, num_layers, dropout_rate, concat_features)
    print(f"Model created with {model.rnn_layers} layers and {model.rnn_hidden_size} hidden size using {model.rnn_type}.")
    os.makedirs(output_dir, exist_ok=True)
    joblib_model_filename = "rnn_model.joblib"
    joblib_model_path = os.path.join(output_dir, joblib_model_filename)
    try:
        joblib.dump(model, joblib_model_path)
        print(f"Model saved to {joblib_model_path} using joblib.")
    except Exception as e:
        print(f"Error saving model with joblib: {e}")

if __name__ == "__main__":
    main()