import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, out_dim, num_layers=2):
        super(Generator, self).__init__()
        # LSTM layer to capture the sequential, temporal dynamics of market data
        self.lstm = nn.LSTM(input_size=z_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        # Linear layer maps the LSTM hidden state to our output dimension (1 for price return)
        self.linear = nn.Linear(hidden_dim, out_dim)
        # Tanh activation squashes the output between [-1, 1], matching our scaled real data
        self.activation = nn.Tanh()
        
    def forward(self, z):
        # z shape: (batch_size, seq_len, z_dim)
        lstm_out, _ = self.lstm(z)
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        out = self.linear(lstm_out)
        out = self.activation(out)
        # out shape: (batch_size, seq_len, out_dim)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(Discriminator, self).__init__()
        # Bidirectional LSTM to process the sequence both forward and backward in time,
        # providing deeper context to classify complex market patterns.
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        # Since it's bidirectional, the hidden state size is doubled (hidden_dim * 2)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Outputs probability of being Real (1) or Fake (0)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # We take the output of the LSTM at the very last time step to make the final classification
        # lstm_out[:, -1, :] shape: (batch_size, hidden_dim * 2)
        out = self.linear(lstm_out[:, -1, :])
        return out
