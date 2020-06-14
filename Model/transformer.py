import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class Encoder_trans(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_heads, pf_dim, dropout=0.5, time_length=53):
        super().__init__()
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.pos_emb = nn.Embedding(time_length, hidden_size)

        self.layers = nn.ModuleList([EncoderLayer(hidden_size, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))

    def forward(self, input_x):
        batch_size = input_x.shape[0]
        enc_len = input_x.shape[1]

        # positonalだけは置き換えの可能性あり
        pos = torch.arange(0, enc_len).unsqueeze(0).repeat(batch_size, 1)
        x = self.dropout((self.input_emb(input_x) * self.scale) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder_trans(nn.Module):
    def __init__(self, input_size, hid_size, n_layers, n_heads, pf_dim, dropout=0.5, time_length=53):
        super().__init__()
        self.input_emb = nn.Linear(input_size, hid_size)
        self.pos_emb = nn.Embedding(time_length, hid_size)
        self.layers = nn.ModuleList([DecoderLayer(hid_size, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_size]))

    def forward(self, dec_x, enc_x, dec_mask=None, enc_mask=None):
        # dec_x = [batch size, trg len]
        # enc_x = [batch size, src len, hid dim]
        # dec_mask = [batch size, trg len]
        # enc_mask = [batch size, src len]
        batch_size = dec_x.shape[0]
        trg_len = dec_x.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1)
        trg = self.dropout((self.input_emb(dec_x) * self.scale) + self.pos_emb(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_x, dec_mask, enc_mask)
        output = self.fc_out(trg)
        return output, attention


# src_maskというoadかどうかを判定した行列を用いる
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # make_src_maskは必要ない
    def mask_trg_mask(self, trg):
        trg_len = trg.shape[1]
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()
        return trg_mask

    def forward(self, enc_i, dec_i):
        src_mask = None

        # batch, 1, len, len
        trg_mask = self.mask_trg_mask(dec_i)
        enc_ = self.encoder(enc_i)
        output, attention = self.decoder(dec_i, enc_, trg_mask, None)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_size, n_heads, pf_dim, dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_size)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_size)
        self.ff_layer_norm = nn.LayerNorm(hid_size)
        # 1つめ
        self.self_attention = MultiHeadAttentionLayer(hid_size, n_heads, dropout)
        # 2つめ
        self.encoder_attention = MultiHeadAttentionLayer(hid_size, n_heads, dropout)
        # 3つめ
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_size, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    # trg, enc_src, trg_mask, src_mask
    def forward(self, dec_x, enc_x, dec_mask=None, enc_mask=None):
        # self-attention
        _trg, _ = self.self_attention(dec_x, dec_x, dec_x, dec_mask)
        trg = self.self_attn_layer_norm(dec_x + self.dropout(_trg))

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_x, enc_x, enc_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # 3つめ
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, pf_dim, dropout=0.5):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttentionLayer(hidden_size, n_heads, dropout)
        self.positionwise_f = PositionwiseFeedforwardLayer(hidden_size, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x):
        _input_x, _ = self.self_attention(input_x, input_x, input_x)
        x = self.self_attn_layer_norm(input_x + self.dropout(_input_x))

        _x = self.positionwise_f(x)
        x = self.ff_layer_norm(x + self.dropout(_x))
        return x

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)
        self.fc_o = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_size]))

    def forward(self, query, key, value, mask=None):
        # q,k,v = batch, len, hidden_size
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # batch, n_head, len, dim
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)

        # batch, len, n_heads, dimmmesnion
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_size)
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_size, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_size, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x batch, seq, hid ⇒ pf_dim
        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, hid dim]
        x = self.fc_2(x)
        return x