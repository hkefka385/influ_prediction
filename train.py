import torch
import torch.nn as nn
import numpy as np
import random
from Model.lstm import LSTM, biLSTM
from Model.gru import GRU, biGRU
from Model.seq2seq import Encoder, Decoder, Seq2Seq
from Model.seq2seq_attn import Encoder_attn, Decoder_attn, Seq2Seq_attn, Attention
from Model.seq2seq_attn_v2 import Encoder_attn2, Decoder_attn2, Seq2Seq_attn2, Attention_attn2
from Model.transformer import Transformer, Encoder_trans, Decoder_trans


def train_lstm(data_i, data_o, validation_i, validation_o, hidden_size, num_layers, epochs, lr=0.001, model_n = 'lstm', opt='Adam',
               loss_f='mse', val_interval=3, shuffle = True, bidirectional = False, batch_n = 1, clip = 1.):
    input_size = data_i.shape[2]
    seq_len = data_i.shape[1]
    if model_n == 'lstm':
        if bidirectional == True:
            model = biLSTM(input_size, hidden_size, num_layers, seq_len)
        else:
            model = LSTM(input_size, hidden_size, num_layers, seq_len)
    elif model_n == 'gru':
        if bidirectional == True:
            model = biGRU(input_size, hidden_size, num_layers, seq_len)
        else:
            model = GRU(input_size, hidden_size, num_layers, seq_len)
    model.apply(_init_weights)
    loss_func = _loss(loss_f)
    optimizer = _opt(opt, lr, model)

    iter_ = len(data_i) // batch_n
    # train
    model.train()
    for i in range(epochs):
        train_input, train_output = _shuffle(data_i, data_o, shuffle)
        for j in range(iter_):
            train_i = train_input[batch_n*j:batch_n*(j+1)]
            train_o =train_output[batch_n*j:batch_n*(j+1)]

            optimizer.zero_grad()
            #model.init_hidden(batch_n = batch_n)
            pred = model(train_i)
            loss = loss_func(pred, train_o)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        if i % val_interval == 1:
            loss = 0
            with torch.no_grad():
                for val_i, val_o in zip(validation_i, validation_o):
                    #model.init_hidden()
                    pred = model(val_i)
                    loss += loss_func(pred, val_o)

            print('Number of epochs: %d' %(i+1))
            print('loss: %f' % loss)
    return model

def train_seq(data_i, data_o, validation_i, validation_o, hidden_size, num_layers, epochs, lr=0.001, model_n = 'lstm', opt='Adam',
               loss_f='mse', val_interval=3, shuffle = True, bidirectional = False, batch_n = 1, clip = 1., dropout = 0.5, teacher_forcing_ratio = 0.5):
    input_size = data_i.shape[2]
    if bidirectional == True:
        enc = Encoder(input_size, hidden_size, num_layers, model_n, dropout = dropout)
        dec = Decoder(input_size, hidden_size, num_layers, model_n, dropout = dropout)
        model = Seq2Seq(enc, dec)
    else:
        enc = Encoder(input_size, hidden_size, num_layers, model_n, dropout = dropout)
        dec = Decoder(input_size, hidden_size, num_layers, model_n, dropout = dropout)
        model = Seq2Seq(enc, dec)

    model.apply(_init_weights)
    loss_func = _loss(loss_f)
    optimizer = _opt(opt, lr, model)

    iter_ = len(data_i) // batch_n
    # train
    model.train()
    for i in range(epochs):
        train_input, train_output = _shuffle(data_i, data_o, shuffle)
        for j in range(iter_):
            train_i = train_input[batch_n*j:batch_n*(j+1)]
            train_o =train_output[batch_n*j:batch_n*(j+1)]

            optimizer.zero_grad()
            pred = model(train_i, train_o, teacher_forcing_ratio = teacher_forcing_ratio)
            loss = loss_func(pred, train_o)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        if i % val_interval == 1:
            loss = 0
            with torch.no_grad():
                for val_i, val_o in zip(validation_i, validation_o):
                    val_i = val_i.unsqueeze(0)
                    val_o = val_o.unsqueeze(0)
                    pred = model(val_i, val_o, teacher_forcing_ratio =0.0)
                    loss += loss_func(pred, val_o)

            print('Number of epochs: %d' %(i+1))
            print('loss: %f' % loss)
    return model

def train_seq_attn(data_i, data_o, validation_i, validation_o, en_hidden_size, de_hidden_size,num_layers, epochs, lr=0.001, opt='Adam',
               loss_f='mse', val_interval=3, shuffle = True, batch_n = 10, clip = 1., dropout = 0.5, teacher_forcing_ratio = 0.5):
    input_size = data_i.shape[2]

    enc = Encoder_attn(input_size, en_hidden_size, num_layers, dropout = dropout)
    attn = Attention(en_hidden_size, de_hidden_size)
    dec = Decoder_attn(input_size,en_hidden_size, de_hidden_size, num_layers, attn, dropout = dropout)
    model = Seq2Seq_attn(enc, dec)

    model.apply(_init_weights)
    loss_func = _loss(loss_f)
    optimizer = _opt(opt, lr, model)

    iter_ = len(data_i) // batch_n
    # train
    model.train()
    for i in range(epochs):
        train_input, train_output = _shuffle(data_i, data_o, shuffle)
        for j in range(iter_):
            train_i = train_input[batch_n*j:batch_n*(j+1)]
            train_o =train_output[batch_n*j:batch_n*(j+1)]

            optimizer.zero_grad()
            pred = model(train_i, train_o, teacher_forcing_ratio = teacher_forcing_ratio)
            loss = loss_func(pred, train_o)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        if i % val_interval == 1:
            loss = 0
            with torch.no_grad():
                for val_i, val_o in zip(validation_i, validation_o):
                    val_i = val_i.unsqueeze(0)
                    val_o = val_o.unsqueeze(0)
                    pred = model(val_i, val_o, teacher_forcing_ratio =0.0)
                    loss += loss_func(pred, val_o)

            print('Number of epochs: %d' %(i+1))
            print('loss: %f' % loss)
    return model

def train_seq_attn2(data_i, data_o, validation_i, validation_o, en_hidden_size, de_hidden_size, epochs, lr=0.001, opt='Adam',
               loss_f='mse', val_interval=3, shuffle = True, batch_n = 1, clip = 1., dropout = 0.5, teacher_forcing_ratio = 0.5):
    input_size = data_i.shape[2]
    enc = Encoder_attn2(input_size, en_hidden_size, dropout = dropout)
    attn = Attention_attn2(en_hidden_size, de_hidden_size)
    dec = Decoder_attn2(input_size,en_hidden_size, de_hidden_size, attn, dropout = dropout)
    model = Seq2Seq_attn2(enc, dec)

    model.apply(_init_weights)
    loss_func = _loss(loss_f)
    optimizer = _opt(opt, lr, model)

    iter_ = len(data_i) // batch_n
    # train
    model.train()
    for i in range(epochs):
        train_input, train_output = _shuffle(data_i, data_o, shuffle)
        for j in range(iter_):
            train_i = train_input[batch_n*j:batch_n*(j+1)]
            train_o =train_output[batch_n*j:batch_n*(j+1)]

            optimizer.zero_grad()
            pred = model(train_i, train_o, teacher_forcing_ratio = teacher_forcing_ratio)
            loss = loss_func(pred, train_o)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        if i % val_interval == 1:
            loss = 0
            with torch.no_grad():
                for val_i, val_o in zip(validation_i, validation_o):
                    val_i = val_i.unsqueeze(0)
                    val_o = val_o.unsqueeze(0)
                    pred = model(val_i, val_o, teacher_forcing_ratio =0.0)
                    loss += loss_func(pred, val_o)

            print('Number of epochs: %d' %(i+1))
            print('loss: %f' % loss)
    return model

def train_transformer(data_i, data_o, validation_i, validation_o, hidden_size, enc_layers, dec_layers, n_heads, en_pf_size, de_pf_size, epochs, lr=0.001, opt='Adam',
               loss_f='mse', val_interval=3, shuffle = True, batch_n = 1, clip = 1., dropout = 0.1):
    input_size = data_i.shape[2]
    seq_len = data_i.shape[1]
    enc = Encoder_trans(input_size, hidden_size*n_heads, enc_layers, n_heads, en_pf_size, dropout = dropout)
    dec = Decoder_trans(input_size, hidden_size*n_heads, dec_layers, n_heads, de_pf_size, dropout = dropout)
    model = Transformer(enc, dec)

    model.apply(_init_weights)
    loss_func = _loss(loss_f)
    optimizer = _opt(opt, lr, model)

    iter_ = len(data_i) // batch_n
    # train
    model.train()
    for i in range(epochs):
        train_input, train_output = _shuffle(data_i, data_o, shuffle)
        for j in range(iter_):
            train_i = train_input[batch_n*j:batch_n*(j+1)]
            train_o =train_output[batch_n*j:batch_n*(j+1)]
            optimizer.zero_grad()
            #model.init_hidden(batch_n = batch_n)

            train_o_i = torch.cat((train_i[:,-1:,:], train_o[:,:-1,:]), dim = 1)
            pred, _ = model(train_i, train_o_i)
            loss = loss_func(pred, train_o)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        if i % val_interval == 1:
            with torch.no_grad():
                future_len = validation_o.shape[1]
                val_o_i = torch.zeros(validation_o.shape)
                val_o_i[:,0,:] = validation_i[:,-1,:]
                for val_iter in range(future_len-1):
                    pred, _ = model(validation_i, val_o_i)
                    val_o_i[:, val_iter+1, :] = pred[:, val_iter, :]
                loss = loss_func(val_o_i, validation_o)
            print('Number of epochs: %d' %(i+1))
            print('loss: %f' % loss)
    return model


def _init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def _shuffle(data_i, data_o,shuffle):
    if shuffle == True:
        shuffle_i = torch.randperm(data_i.size()[0])
        train_i_ = data_i[shuffle_i]
        train_o_ = data_o[shuffle_i]
        return train_i_, train_o_
    else:
        return data_i, data_o

def _loss(loss_f):
    if loss_f == 'mse':
        return nn.MSELoss()
    else:
        return nn.MSELoss()


def _opt(opt, lr, model):
    if opt == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)