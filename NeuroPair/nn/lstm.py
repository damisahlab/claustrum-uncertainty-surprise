import torch
import torch.nn as nn
import torch.nn.functional as F

class RnnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, num_layers, dropout):
        super(RnnLSTM, self).__init__()  # Correct the superclass name

        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size 
        self.dropout = dropout

        # Define an LSTM layer
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=self.dropout if num_layers > 1 else 0,   
            batch_first=True
        )

        # Define fully connected layers
        self.output_layer1 = nn.Linear(self.hidden_size, 128)
        self.output_layer2 = nn.Linear(128, self.input_size)
        self.relu = nn.ReLU()

    def forward(self, x, h_0, c_0): 
        x, (hidden_n, cell_n) = self.rnn(x, (h_0, c_0))