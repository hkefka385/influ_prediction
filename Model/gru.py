import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, dropout=0.5):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)


    def forward(self, x):
        gru_out, _ = self.gru(x.view(-1, self.seq_len, self.input_size))
        output = self.fc(gru_out[:, -1, :].view(-1, self.hidden_size))
        return output

class biGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len,dropout = 0.5):
        super(biGRU, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.se
        q_len = seq_len
        self.bigru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, input_size)


    def forward(self, x):
        gru_out, _ = self.bigru(x.view(-1, self.seq_len,self.input_size))
        output = self.fc(gru_out[:,-1,:].view(-1, self.hidden_size*2))
        return output