#!/usr/bin/env python3

import time
import math
import subprocess
import sys
import random
import webbrowser
import numpy as np
from datetime import datetime
from dateutil import tz
import pytz
import os, sys
from influxdb import InfluxDBClient
import operator
import copy
from collections import Counter

from scipy.stats import norm
from scipy.special import softmax
import pandas as pd

import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_absolute_error
import torch



# get_ipython().system(' pip install tsfel # installing TSFEL for feature extraction')
def str2bool(v):
  return v.lower() in ("true", "1", "https", "load")

def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        raise ValueError('invalid mac address')
    return int(res.group(0).replace(':', ''), 16)

def int_to_mac(macint):
    # if type(macint) != int:
    #     raise ValueError('invalid integer')
    newint = int(macint)
    return ':'.join(['{}{}'.format(a, b)
                     for a, b
                     in zip(*[iter('{:012x}'.format(newint))]*2)])

# This function converts the time string to epoch time xxx.xxx (second.ms).
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def local_time_epoch(time, zone):
    local_tz = pytz.timezone(zone)
    localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    local_dt = local_tz.localize(localTime, is_dst=None)
    # utc_dt = local_dt.astimezone(pytz.utc)
    epoch = local_dt.timestamp()
    # print("epoch time:", epoch) # this is the epoch time in seconds, times 1000 will become epoch time in milliseconds
    # print(type(epoch)) # float
    return epoch



# This function converts the epoch time xxx.xxx (second.ms) to time string.
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def epoch_time_local(epoch, zone):
    local_tz = pytz.timezone(zone)
    time = datetime.fromtimestamp(epoch).astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return time

# This function converts the grafana URL time to epoch time. For exmaple, given below URL
# https://sensorweb.us:3000/grafana/d/OSjxFKvGk/caretaker-vital-signs?orgId=1&var-mac=b8:27:eb:6c:6e:22&from=1612293741993&to=1612294445244
# 1612293741993 means epoch time 1612293741.993; 1612294445244 means epoch time 1612294445.244
def grafana_time_epoch(time):
    return time/1000

def influx_time_epoch(time):
    return time/10e8

def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp):
    if influx['ip'] == '127.0.0.1' or influx['ip'] == 'localhost':
        client = InfluxDBClient(influx['ip'], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])
    else:
        client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])

    # client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=True)
    query = 'SELECT "' + data_name + '" FROM "' + table_name + '" WHERE "location" = \''+unit+'\' AND time >= '+ str(int(start_timestamp*10e8))+' AND time < '+str(int(end_timestamp*10e8))
    # query = 'SELECT last("H") FROM "labelled" WHERE ("location" = \''+unit+'\')'

    # print(query)
    result = client.query(query)
    # print(result)

    points = list(result.get_points())
    values =  list(map(operator.itemgetter(data_name), points))
    times  =  list(map(operator.itemgetter('time'),  points))
    # print(times)
    # times = [local_time_epoch(item[:-1], "UTC") for item in times] # convert string time to epoch time
    # print(times)

    data = values #np.array(values)
    # print(data, times)
    return data, times

def load_data_file(data_file):
    if data_file.endswith('.csv'):
        data_set = pd.read_csv(data_file).to_numpy()
    elif data_file.endswith('.npy'):
        data_set = np.load(data_file)
    return data_set


def calc_mae(gt, pred):
    return mean_absolute_error(gt,pred)
    
# list1: label; list2: prediction
def plot_2vectors(label, pred, name, titled):
    list1 = label
    list2 = pred
    if len(list2.shape) == 2:
        mae = calc_mae(list1[:,0], list2[:,0])
    else:
        mae = calc_mae(list1, list2)

    # zipped_lists = zip(list1, list2)
    # sorted_pairs = sorted(zipped_lists)

    # tuples = zip(*sorted_pairs)
    # list1, list2 = np.array([ list(tuple) for tuple in  tuples])

    # print(list1.shape)
    # print(list2.shape)
    
    sorted_id = sorted(range(len(list1)), key=lambda k: list1[k])

    plt.clf()
    plt.text(0,np.min(list2),f'MAE={mae}')

    # plt.plot(range(num_rows), list2, label=name + ' prediction')
    plt.scatter(np.arange(list2.shape[0]),list2[sorted_id],s = 1, alpha=0.5,label=f'{name} prediction', color='blue')

    plt.scatter(np.arange(list1.shape[0]),list1[sorted_id],s = 1, alpha=0.5,label=f'{name} label', color='red')

    # plt.plot(range(num_rows), list1, 'r.', label=name + ' label')
    plt.title(titled)
    plt.xlabel('Number of Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{name}.png')
    print(f'Saved plot to {name}.png')
    plt.show()

def find_best_device():
    if not torch.cuda.is_available():
        return "cpu"
    # elif not args.cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #     return "cpu"
    import nvidia_smi #pip3 install nvidia-ml-py3

    nvidia_smi.nvmlInit()
    best_gpu_id = 0
    best_free = 0 
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
        if info.free > best_free:
            best_free = info.free
            best_gpu_id = i
    nvidia_smi.nvmlShutdown()
    # print(f"Best GPU to use is cuda:{best_gpu_id}!")
    return f"cuda:{best_gpu_id}"

## Transformations

def create_individual_transform_dataset(X, transform_funcs, other_labels=None, multiple=1, is_transform_func_vectorized=True, verbose=1):
    label_depth = len(transform_funcs)
    transform_x = []
    transform_y = []
    other_y = []
    if is_transform_func_vectorized:
        for _ in range(multiple):
            
            transform_x.append(X)
            ys = np.zeros((len(X), label_depth), dtype=int)
            transform_y.append(ys)
            if other_labels is not None:
                other_y.append(other_labels)

            for i, transform_func in enumerate(transform_funcs):
                if verbose > 0:
                    print(f"Using transformation {i} {transform_func}")
                transform_x.append(transform_func(X))
                ys = np.zeros((len(X), label_depth), dtype=int)
                ys[:, i] = 1
                transform_y.append(ys)
                if other_labels is not None:
                    other_y.append(other_labels)
        if other_labels is not None:
            return np.concatenate(transform_x, axis=0), np.concatenate(transform_y, axis=0), np.concatenate(other_y, axis=0)
        else:
            return np.concatenate(transform_x, axis=0), np.concatenate(transform_y, axis=0), 
    else:
        for _ in range(multiple):
            for i, sample in enumerate(X):
                if verbose > 0 and i % 1000 == 0:
                    print(f"Processing sample {i}")
                    gc.collect()
                y = np.zeros(label_depth, dtype=int)
                transform_x.append(sample)
                transform_y.append(y)
                if other_labels is not None:
                    other_y.append(other_labels[i])
                for j, transform_func in enumerate(transform_funcs):
                    y = np.zeros(label_depth, dtype=int)
                    # transform_x.append(sample)
                    # transform_y.append(y.copy())

                    y[j] = 1
                    transform_x.append(transform_func(sample))
                    transform_y.append(y)
                    if other_labels is not None:
                        other_y.append(other_labels[i])
        if other_labels is not None:
            np.stack(transform_x), np.stack(transform_y), np.stack(other_y)
        else:
            return np.stack(transform_x), np.stack(transform_y)
        

def map_multitask_y(y, output_tasks):
    multitask_y = {}
    for i, task in enumerate(output_tasks):
        multitask_y[task] = y[:, i]
    return multitask_y