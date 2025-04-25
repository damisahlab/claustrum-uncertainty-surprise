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
        # Pass input through the RNN layer
        x, (hidden_n, cell_n) = self.rnn(x, (h_0, c_0))

        # # Flatten the output across the sequence dimension for input to fully connected layers
        # x = x.reshape(-1, self.hidden_size)  # Correctly flatten x for FC layers

        # # Pass through the first fully connected layer
        # x = self.relu(self.output_layer1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout appropriately

        # # Pass through the second fully connected layer
        # x = self.output_layer2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)

        # # Reshape x to match expected output dimensions
        # x = x.reshape(-1, self.seq_len, self.input_size)

        # # Obtain the last output and apply mean across the sequence
        # # x = x[:, -1].mean(dim=1, keepdim=True)  # Simplified the mean calculation

        # return x, hidden_n, cell_n


