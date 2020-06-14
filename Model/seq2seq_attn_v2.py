import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class Encoder_attn2(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # hiddenがdecoderにわたすよう
        # outputsがattention用
        # outputs = batch * src * (hid * 2)
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        # outputs, hidden = self.rnn(x)

        # batch * hidden_size
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention_attn2(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        # encoderからのoutputs + decoder_
        self.attn = nn.Linear(enc_dim * 2 + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        en_len = encoder_outputs.shape[1]

        # hidden_state をen_len分繰り返し
        hidden = hidden.unsqueeze(1).repeat(1, en_len, 1)

        # hidden: batch * en_len * dec_dim
        # encoder_outputs: batch * en_len * (enc_dim * 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # batch * en_len
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


# attentionを組み込み
class Decoder_attn2(nn.Module):
    def __init__(self, input_size, enc_hidden_size, dec_hidden_size, attention, dropout=0.5):
        super().__init__()
        self.output_size = input_size
        self.embedding_dec = nn.Linear(input_size, dec_hidden_size)
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.hidden_size = dec_hidden_size
        self.attention = attention
        self.rnn = nn.GRU((enc_hidden_size * 2) + dec_hidden_size, dec_hidden_size, batch_first=True)

        self.fc = nn.Linear((enc_hidden_size * 2) + dec_hidden_size + dec_hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs):
        # x: batch * 1 * 1

        # attention_: batch * en_len

        # x = x.clone()でなぜか解決？
        embed_x = self.dropout(self.embedding_dec(x.clone()))

        attention_ = self.attention(hidden, encoder_outputs)
        # batch * 1 * en_len
        attention_ = attention_.unsqueeze(1)
        # batch * 1 * (enc_dim * 2)
        weighted = torch.bmm(attention_, encoder_outputs)
        rnn_input = torch.cat((embed_x, weighted), dim=2)
        # rnn_input = torch.cat((x_, weighted), dim = 2)
        # batch * 1 * (enc_dim * 2 + 1)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = output.view(-1, self.hidden_size)
        weighted = weighted.view(-1, self.enc_hidden_size * 2)
        prediction = self.fc(torch.cat((output, weighted, embed_x.squeeze(1)), dim=1))

        return prediction, hidden.squeeze(0)


class Seq2Seq_attn2(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, output, teacher_forcing_ratio=0.5):
        batch_size = output.shape[0]
        seq_len = output.shape[1]
        output_size = self.decoder.output_size
        outputs = torch.zeros(batch_size, seq_len, output_size)
        encoder_outputs, hidden = self.encoder(input)
        de_input = input[:, -2:-1, :]
        for t in range(seq_len):
            de_output, hidden = self.decoder(de_input, hidden, encoder_outputs)
            de_output =  de_output.unsqueeze(1)
            outputs[:, t:t+1, :] = de_output
            teacher_force = random.random() < teacher_forcing_ratio
            de_input = output[:, t:t + 1, :] if teacher_force else outputs[:, t:t + 1, :]

        return outputs