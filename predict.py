import torch
import torch.nn as nn
import numpy as np

def predict(model, test_data):
    predict_ = []
    model.eval()
    for test_i in test_data:
        with torch.no_grad():
            predict_.append(model(test_i)[-1].item())
    return predict_

def predict_seq(model, test_input, test_output):
    predict_ = []
    model.eval()
    with torch.no_grad():
        for test_i, test_o in zip(test_input, test_output):
            test_i = test_i.unsqueeze(0)
            test_o = test_o.unsqueeze(0)
            predict_.append(model(test_i, test_o, teacher_forcing_ratio = 0.0).view(test_o.size()[1]).numpy())
    return np.array(predict_)

def predict_trans(model, test_input, test_output):
    model.eval()
    with torch.no_grad():
        future_len = test_output.shape[1]
        test_o_i = torch.zeros(test_output.shape)
        test_o_i[:,0,:] = test_input[:,-1,:]
        for test_iter in range(future_len - 1):
            pred, _ = model(test_input, test_o_i)
            test_o_i[:, test_iter + 1, :] = pred[:, test_iter, :]

    return np.array(test_o_i)
