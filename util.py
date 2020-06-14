import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def train_test_split(input_d, val_n, test_n, time_window, future_step = 1, train_n = False):
    train_lab = []
    train_inp = []
    valid_lab = []
    valid_inp = []
    test_lab = []
    test_inp = []
    if train_n:
        for i in range(len(input_d) - val_n - test_n- train_n, len(input_d) - val_n - test_n):
            train_lab.append(input_d[i])
            train_inp.append(input_d[i - time_window - future_step+1:i-future_step+1])
    else:
        for i in range(time_window+future_step-1, len(input_d) - val_n - test_n):
            train_lab.append(input_d[i])
            train_inp.append(input_d[i-time_window-future_step+1:i-future_step+1])

    for i in range(len(input_d) - val_n - test_n, len(input_d) - test_n):
        valid_lab.append(input_d[i])
        valid_inp.append(input_d[i - time_window - future_step+1:i-future_step+1])
    for i in range(len(input_d) - test_n, len(input_d)):
        test_lab.append(input_d[i])
        test_inp.append(input_d[i - time_window - future_step+1:i-future_step+1])
    return [np.array(train_inp), np.array(train_lab)], [np.array(valid_inp), np.array(valid_lab)], [np.array(test_inp), np.array(test_lab)]

def train_test_split_seq(input_d, val_n, test_n, time_window, future_step = 1, train_n = False):
    train_lab = []
    train_inp = []
    valid_lab = []
    valid_inp = []
    test_lab = []
    test_inp = []
    if train_n:
        for i in range(len(input_d) - val_n - test_n- train_n, len(input_d) - val_n - test_n):
            train_lab.append(input_d[i-future_step+1:i+1])
            train_inp.append(input_d[i - time_window - future_step+1:i-future_step+1])
    else:
        for i in range(time_window+future_step-1, len(input_d) - val_n - test_n):
            train_lab.append(input_d[i-future_step+1:i+1])
            train_inp.append(input_d[i-time_window-future_step+1:i-future_step+1])

    for i in range(len(input_d) - val_n - test_n, len(input_d) - test_n):
        valid_lab.append(input_d[i-future_step+1:i+1])
        valid_inp.append(input_d[i - time_window - future_step+1:i-future_step+1])
    for i in range(len(input_d) - test_n, len(input_d)):
        test_lab.append(input_d[i-future_step+1:i+1])
        test_inp.append(input_d[i - time_window - future_step+1:i-future_step+1])
    return [np.array(train_inp), np.array(train_lab)], [np.array(valid_inp), np.array(valid_lab)], [np.array(test_inp), np.array(test_lab)]

def normalize_func(data ,val_n, test_n, method = 'minmax'):
    if method == 'minmax':
        minmax = MinMaxScaler()
        minmax.fit(data[:-(val_n + test_n)])
        data = minmax.fit_transform(data)
    return data, minmax
