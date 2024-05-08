import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, rnn_type='LSTM'):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        if hasattr(self, 'rnn') and isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            # One time step
            out, (hn, cn) = self.rnn(x, (h0, c0))
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            out, hn = self.rnn(x, h0)
        
        out = self.fc(out[:, -1, :])  # Taking last time step's output
        return out
