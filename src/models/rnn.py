import torch
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
                 num_layers=1, dropout_rate=0.2):
        super(RNNSurvivalPredictor, self).__init__()

        self.rnn_layers = num_layers
        self.rnn_hidden_size = hidden_size
        self.rnn_type = rnn_type

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                              batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                               batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(site_features_size + hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, site_features, sequence_lengths):
        batch_size = x.size(0)
        h0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(x.device)

        packed_input = rnn_utils.pack_padded_sequence(
                        x,
                        sequence_lengths.cpu().long(),
                        batch_first=True,
                        enforce_sorted=False)

        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(x.device)
            packed_output, (hn, cn) = self.rnn(packed_input, (h0, c0))
        else:
            packed_output, hn = self.rnn(packed_input, h0)

        last_hidden_state = hn[-1]
        concatenated_features = torch.cat((last_hidden_state, site_features), dim=1)
        ffn_input_dropped = self.dropout(concatenated_features)
        ffn_hidden_output = self.activation_ffn(self.ffn_layer1(ffn_input_dropped))
        ffn_hidden_dropped = self.dropout(ffn_hidden_output)
        predicted_value_raw = self.ffn_output_layer(ffn_hidden_dropped)

        clamped_output = torch.clamp(predicted_value_raw, 0, 100)
        return clamped_output
