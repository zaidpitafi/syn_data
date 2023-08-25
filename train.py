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
from scipy.signal import find_peaks
from ResNet import *

train_dataset = np.load("../DataDemo-main/simu_20000_0.5_90_140_train_cycle_peaks.npy")

X_train = train_dataset[:, :-6]
Y_train = train_dataset[:, -2].reshape(-1,1)

test_dataset = np.load("../DataDemo-main/simu_10000_0.5_141_178_test_cycle_peaks.npy")
X_test = test_dataset[:, :-6]
Y_test = test_dataset[:, -2].reshape(-1,1)

X_train_mean = np.mean(X_train, axis = 1)
X_train_mean = np.reshape(X_train_mean, (-1,1))
X_train_std = np.std(X_train, axis = 1)
X_train_std = np.reshape(X_train_std, (-1,1))
X_train = (X_train - X_train_mean)/X_train_std

X_test_mean = np.mean(X_test, axis = 1)
X_test_mean = np.reshape(X_test_mean, (-1,1))
X_test_std = np.std(X_test, axis = 1)
X_test_std = np.reshape(X_test_std, (-1,1))
X_test = (X_test - X_test_mean)/X_test_std

X_train = np.expand_dims(X_train, axis=-2)
X_test = np.expand_dims(X_test, axis=-2)

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def my_train_pred():
    epoches = 50
    lr = 0.001
    batch_size = 32
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float()) 
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float()) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print(X_train.shape)
    net = ResNet(c_in=1, c_out=1).to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    history = []
    for epoch in range(epoches):
        net.train()
        total_mae = 0
        total_loss = 0
        number = 0
        for index, (batch_x, batch_y) in enumerate(train_loader):
            optim.zero_grad()
            out = net(batch_x.to(DEVICE))
            loss = criterion(out, batch_y.to(DEVICE))
            loss.backward()
            mae = mean_absolute_error(out.detach().cpu(), batch_y)
            total_mae += mae
            l = loss.item()
            total_loss += l
            optim.step()
            number += 1
        history.append([total_mae / number, total_loss / number])
        net.eval()
        test_mae = 0
        counter = 0
        with torch.no_grad():
            for index, (batch_x, batch_y) in enumerate(test_loader):
                out = net(batch_x.to(DEVICE))
                mae = mean_absolute_error(out.detach().cpu(), batch_y)
                test_mae += mae
                counter += 1
        print("mae:", epoch, test_mae / counter, total_mae / number)       
        torch.save(net.state_dict(), f"./results/ckpt_best_{epoch}_{test_mae/counter:.2f}.pt")
        
if __name__ == '__main__':
    my_train_pred()