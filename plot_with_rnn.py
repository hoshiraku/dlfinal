#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 05:10:48 2018

@author: hoshiraku
"""

import json
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 40
configs =[]
learning_curves = []

t_idx = np.arange(1, num_epochs+1)

# data with different epochs
fn_1 = './data/results/rnn/config_rnn_1.json'

with open(fn_1, 'r') as fh:
    tmp = json.load(fh)
    config_1 = tmp['config']
    configs.append(config_1)
    learning_curve_1 = tmp['learning_curve']
    learning_curves.append(learning_curve_1)
    

fn_2 = './data/results/rnn/config_rnn_2.json'

with open(fn_2, 'r') as fh:
    tmp = json.load(fh)
    config_2 = tmp['config']
    configs.append(config_2)
    learning_curve_2 = tmp['learning_curve']
    learning_curves.append(learning_curve_2)

fn_3 = './data/results/rnn/config_rnn_3.json'

with open(fn_3, 'r') as fh:
    tmp = json.load(fh)
    config_3 = tmp['config']
    configs.append(config_3)
    learning_curve_3 = tmp['learning_curve']
    learning_curves.append(learning_curve_3)

fn_4 = './data/results/rnn/config_rnn_4.json'

with open(fn_4, 'r') as fh:
    tmp = json.load(fh)
    config_4 = tmp['config']
    configs.append(config_4)
    learning_curve_4 = tmp['learning_curve']
    learning_curves.append(learning_curve_4)
    
fn_5 = './data/results/rnn/config_rnn_5.json'

with open(fn_5, 'r') as fh:
    tmp = json.load(fh)
    config_5 = tmp['config']
    configs.append(config_5)
    learning_curve_5 = tmp['learning_curve']
    learning_curves.append(learning_curve_5)

fig, ax = plt.subplots()
ax.plot(t_idx, learning_curve_1, 'r', label='using first 5 epochs')
ax.plot(t_idx, learning_curve_2, 'g', label='using first 10 epochs')
ax.plot(t_idx, learning_curve_3, 'b', label='using first 20 epochs')
ax.plot(t_idx, learning_curve_4, 'k', label='using first 8 epochs (random)')
ax.plot(t_idx, learning_curve_5, 'm', label='using first 15 epochs (random)')
legend = ax.legend(loc='upper right', shadow=False)
#legend.get_frame().set_facecolor('#00FF00')
plt.title("Learning curves of RNN using different epochs")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")                
plt.show()


# data with different learning rates
# RNN using first 10 epochs
fn_1 = './data/results/rnn/config_rnn_2.json'

with open(fn_1, 'r') as fh:
    tmp = json.load(fh)
    config_1 = tmp['config']
    configs.append(config_1)
    learning_curve_1 = tmp['learning_curve']
    learning_curves.append(learning_curve_1)
    

fn_2 = './data/results/rnn/config_rnn_6.json'

with open(fn_2, 'r') as fh:
    tmp = json.load(fh)
    config_2 = tmp['config']
    configs.append(config_2)
    learning_curve_2 = tmp['learning_curve']
    learning_curves.append(learning_curve_2)

fn_3 = './data/results/rnn/config_rnn_7.json'

with open(fn_3, 'r') as fh:
    tmp = json.load(fh)
    config_3 = tmp['config']
    configs.append(config_3)
    learning_curve_3 = tmp['learning_curve']
    learning_curves.append(learning_curve_3)

fn_4 = './data/results/rnn/config_rnn_8.json'

with open(fn_4, 'r') as fh:
    tmp = json.load(fh)
    config_4 = tmp['config']
    configs.append(config_4)
    learning_curve_4 = tmp['learning_curve']
    learning_curves.append(learning_curve_4)

fig, ax = plt.subplots()
ax.plot(t_idx, learning_curve_1, 'r', label='lr = 0.01')
ax.plot(t_idx, learning_curve_2, 'g', label='lr = 0.005')
ax.plot(t_idx, learning_curve_3, 'b', label='lr = 0.001')
ax.plot(t_idx, learning_curve_4, 'k', label='lr = 0.05')
legend = ax.legend(loc='upper right', shadow=False)
#legend.get_frame().set_facecolor('#00FF00')
plt.title("Learning curves of RNN using different learning rates")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")                
plt.show()


# data with different drop rates
# RNN using first 10 epochs
fn_1 = './data/results/rnn/config_rnn_2.json'

with open(fn_1, 'r') as fh:
    tmp = json.load(fh)
    config_1 = tmp['config']
    configs.append(config_1)
    learning_curve_1 = tmp['learning_curve']
    learning_curves.append(learning_curve_1)
    

fn_2 = './data/results/rnn/config_rnn_9.json'

with open(fn_2, 'r') as fh:
    tmp = json.load(fh)
    config_2 = tmp['config']
    configs.append(config_2)
    learning_curve_2 = tmp['learning_curve']
    learning_curves.append(learning_curve_2)

fn_3 = './data/results/rnn/config_rnn_10.json'

with open(fn_3, 'r') as fh:
    tmp = json.load(fh)
    config_3 = tmp['config']
    configs.append(config_3)
    learning_curve_3 = tmp['learning_curve']
    learning_curves.append(learning_curve_3)

fn_4 = './data/results/rnn/config_rnn_11.json'

with open(fn_4, 'r') as fh:
    tmp = json.load(fh)
    config_4 = tmp['config']
    configs.append(config_4)
    learning_curve_4 = tmp['learning_curve']
    learning_curves.append(learning_curve_4)

fig, ax = plt.subplots()
ax.plot(t_idx, learning_curve_1, 'r', label='drop rate = 0.2')
ax.plot(t_idx, learning_curve_2, 'g', label='drop rate = 0.1')
ax.plot(t_idx, learning_curve_3, 'b', label='drop rate = 0.3')
ax.plot(t_idx, learning_curve_4, 'k', label='drop rate = 0.5')
legend = ax.legend(loc='upper right', shadow=False)
#legend.get_frame().set_facecolor('#00FF00')
plt.title("Learning curves of RNN using different drop rates")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")                
plt.show()