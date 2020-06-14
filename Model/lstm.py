import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len,dropout = 0.5):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        #hidden: (num_layers*bi?) * batch * hidden
        #input: batch * seq_len(52) * input_size(1)
        #lstm_out: batch * seq_len(52) * hidden_size

        lstm_out, _ = self.lstm(x.view(-1, self.seq_len,self.input_size))
        output = self.fc(lstm_out[:,-1,:].view(-1, self.hidden_size))
        return output

class biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, dropout = 0.5):
        super(biLSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.bilstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, input_size)


    def forward(self, x):
        lstm_out, _ = self.bilstm(x.view(-1, self.seq_len,self.input_size))
        output = self.fc(lstm_out[:,-1,:].view(-1, self.hidden_size*2))
        return output