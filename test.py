#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from TCN import *
from utils import *
from ResNet import *

test_dataset = np.load("../DataDemo-main/simu_10000_0.5_141_178_test_cycle_peaks.npy")
X_test = test_dataset[:, :-6]
Y_test = test_dataset[:, -2].reshape(-1,1)
print(X_test.shape)

X_test_mean = np.mean(X_test, axis = 1)
X_test_mean = np.reshape(X_test_mean, (-1,1))
X_test_std = np.std(X_test, axis = 1)
X_test_std = np.reshape(X_test_std, (-1,1))
X_test = (X_test - X_test_mean)/X_test_std

X_test = np.expand_dims(X_test, axis=-2)

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def my_test_pred():
    batch_size = 128
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float()) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    net = ResNet(c_in=1, c_out=1).to(DEVICE)
    checkpoint = torch.load('./results/ckpt_best_25_0.11.pt')
    net.load_state_dict(checkpoint)
    net.eval()
    result = []
    y_true=[]
    with torch.no_grad():
        for index, (batch_x, batch_y) in enumerate(test_loader):
            out = net(batch_x.to(DEVICE))
            out = out.detach().cpu().numpy().squeeze()
            result.extend(out)
            y_true.extend(batch_y.detach().cpu().numpy().squeeze())
    pred = np.asarray(result)
    label = np.asarray(y_true)
    np.save('labels_2', pred)
    plot_2vectors(label,pred, 'S')
    # plot_2vectors(label[:,0],pred[:,0], 'S')
    # plot_2vectors(label[:,1],pred[:,1], 'D')

if __name__ == '__main__':
    my_test_pred()