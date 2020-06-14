import torch
import torch.nn as nn
import random

## Encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_n = 'gru', dropout = 0.5):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        if model_n == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout = dropout, batch_first = True)
            self.hidden_size = hidden_size
        elif model_n == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, batch_first = True)
            self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # hidden = n_layers * n_directions, batch_size, hid_dim
        outputs, hidden = self.rnn(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_n = 'gru', dropout = 0.5):
        super().__init__()
        self.output_size = input_size
        self.input_size = input_size
        self.num_layers = num_layers
        if model_n == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout = dropout, batch_first = True)
            self.hidden_size = hidden_size
            self.fc_out = nn.Linear(hidden_size, self.output_size)
        elif model_n == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, batch_first = True)
            self.hidden_size = hidden_size
            self.fc_out = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        input = x.unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,input, output, teacher_forcing_ratio=0.5):
        batch_size = output.shape[0]
        seq_len = output.shape[1]
        output_size = self.decoder.output_size
        outputs = torch.zeros(batch_size, seq_len, output_size)
        # encoder
        hidden = self.encoder(input)
        #decoder
        de_input = input[:,-1,:]
        for t in range(seq_len):
            de_output, hidden = self.decoder(de_input, hidden)
            outputs[:,t:t+1,:] = de_output
            teacher_force = random.random() < teacher_forcing_ratio
            de_input = output[:,t,:] if teacher_force else outputs[:,t,:]
        return outputs
