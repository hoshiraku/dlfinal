# -*- coding: utf-8 -*-

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(source_dir='./data/final_project'):
    
    configs = []
    learning_curves = []
    
    for fn in glob.glob(os.path.join(source_dir, "*.json")):
        with open(fn, 'r') as fh:
            tmp = json.load(fh)
            configs.append(tmp['config'])
            learning_curves.append(tmp['learning_curve'])
    return(configs, learning_curves)

configs, learning_curves = load_data()

# keep a copy for input and output labels
configs_copy = configs
learning_curves_copy = learning_curves

N = len(configs)
#print(N)
n_epochs = len(learning_curves[0])

configs_df = pd.DataFrame(configs)
learning_curves = np.array(learning_curves)

labels = np.zeros((N, 1))
for i in range(N):
    line = learning_curves[i]
    labels[i][0] = line[-1]
    

inputs = np.zeros((N, 5))
for i in range(N):
    json_in = configs[i]
    inputs[i][0] = json_in['batch_size']
    inputs[i][1] = json_in['log10_learning_rate']
    inputs[i][2] = json_in['log2_n_units_1']
    inputs[i][3] = json_in['log2_n_units_2']
    inputs[i][4] = json_in['log2_n_units_3']

"""    
n_subset=20
t_idx = np.arange(1, n_epochs+1)

[plt.plot(t_idx, lc) for lc in learning_curves[:n_subset]]
plt.title("Subset of learning curves")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")
plt.show()

sorted = np.sort(learning_curves[:, -1])
h = plt.hist(sorted, bins=20)
plt.show()

yvals = np.arange(len(sorted))/float(len(sorted))
plt.plot(sorted, yvals)
plt.title("Empirical CDF")
plt.xlabel("y(x, t=40)")
plt.ylabel("CDF(y)")
plt.show()

all_values = np.sort(learning_curves.flatten())

h = plt.hist(all_values, bins=20)
plt.show()

yvals = np.arange(all_values.shape[0])/all_values.shape[0]
plt.plot(all_values, yvals)
plt.title("Empirical CDF")
plt.xlabel("y(x, t=40)")
plt.ylabel("CDF(y)")
plt.show()

"""
# Pre-processing the data


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random

# Define the model

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.input = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.85)
        self.out = nn.Linear(hidden_size, output_size)
        
    