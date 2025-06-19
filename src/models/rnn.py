from unicodedata import bidirectional
import torch
import click
import os
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
        linear_size (int): Size of the first linear layer output. 
        rnn_type (str): Type of RNN to use ('GRU' or 'LSTM').
        rnn (nn.Module): The recurrent neural network layer (GRU or LSTM).
        activation (nn.ReLU): Activation function for the output.
        linear (nn.Linear): Linear layer for final prediction.
        dropout (nn.Dropout): Dropout layer for regularization.
    Args:
        input_size (int): The number of features in each time step of the input sequence.
        hidden_size (int): The size of the hidden state in the RNN.
        linear_size (int): Size of the first linear layer output. 
        site_features_size (int): The number of static features per site.
        rnn_type (str, optional): The type of RNN to use ('GRU' or 'LSTM'). Defaults to "GRU".
        num_layers (int, optional): Number of recurrent layers. Defaults to 1.
        dropout_rate (float, optional): Dropout probability for regularization. Defaults to 0.2.
    """
    def __init__(self, input_size, hidden_size,linear_size,
                 site_features_size, rnn_type="GRU",
                 num_layers=1, dropout_rate=0.2, concat_features=False):
        super(RNNSurvivalPredictor, self).__init__()

        self.rnn_layers = num_layers
        self.rnn_hidden_size = hidden_size
        self.linear_size = linear_size
        self.rnn_type = rnn_type
        self.concat_features = concat_features
        self.input_linear_size = site_features_size + hidden_size*2 if self.concat_features else hidden_size*2
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        if rnn_type not in ['GRU', 'LSTM']:
            raise ValueError("rnn_type must be either 'GRU' or 'LSTM'")

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=True,
                              batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True,
                               batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        self.linear_sequence =  nn.Sequential(
            nn.Linear(self.input_linear_size, self.linear_size),
            self.activation,
            nn.Linear(self.linear_size, 1),
            self.activation
            )

    def forward(self, sequence, sequence_length, site_features):
        batch_size = sequence.size(0)
        h0 = torch.zeros(self.rnn_layers*2, batch_size, self.rnn_hidden_size).to(sequence.device)

        packed_input = rnn_utils.pack_padded_sequence(
                        sequence,
                        sequence_length.long(),
                        batch_first=True,
                        enforce_sorted=False)

        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.rnn_layers*2, batch_size, self.rnn_hidden_size).to(sequence.device)
            packed_output, (hn, cn) = self.rnn(packed_input, (h0, c0))
        else:
            packed_output, hn = self.rnn(packed_input, h0)
        

        last_hidden_state = torch.cat((hn[-2], hn[-1]), dim=1)
        concatenated_features = torch.cat((last_hidden_state, site_features), dim=1) if self.concat_features else last_hidden_state
        output = self.linear_sequence(concatenated_features)
        output = self.dropout(output)
        output = torch.mul(torch.sigmoid(output),100)
        return output.squeeze()

@click.command()
@click.option('--input_size', type=int, required=True, help='Number of features in each time step of the input sequence')
@click.option('--hidden_size', type=int, required=True, help='Size of the hidden state in the RNN')
@click.option('--linear_size', type=int, required=True, help='Size of the linear layer in the RNN')
@click.option('--site_features_size', type=int, required=True, help='Number of static features per site')
@click.option('--rnn_type', type=click.Choice(['GRU', 'LSTM']), default='GRU', help='Type of RNN to use')
@click.option('--num_layers', type=int, default=1, help='Number of recurrent layers')
@click.option('--dropout_rate', type=float, default=0.2, help='Dropout probability for regularization')
@click.option('--concat_features', type=bool, default=False, help='Concatenate site features with RNN output')
@click.option('--output_path', type=click.Path(exists=False), required=True, help='Path to save the model')
def main(input_size, hidden_size, linear_size, site_features_size, rnn_type, num_layers, dropout_rate, concat_features, output_path):
    '''
    CLI for constructing a RNN based model pipelines.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = RNNSurvivalPredictor(input_size, hidden_size, linear_size, site_features_size, rnn_type, num_layers, dropout_rate, concat_features)
    model = model.to(device)
    print(f"Model created with {model.rnn_layers} layers and {model.rnn_hidden_size} hidden size using {model.rnn_type}.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    config = {
        'input_size': input_size, 
        'hidden_size': hidden_size, 
        'linear_size': linear_size,
        'site_features_size': site_features_size, 
        'rnn_type': rnn_type, 
        'num_layers': num_layers, 
        'dropout_rate': dropout_rate, 
        'concat_features': concat_features
    }
    
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config
        }, output_path)
        print(f"Model saved to {output_path}.")
    except Exception as e:
        print(f"Error saving model {e}")

if __name__ == "__main__":
    main()