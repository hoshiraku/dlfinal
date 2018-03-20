#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 04:09:58 2018

@author: hoshiraku
"""

import json
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 40
configs =[]
learning_curves = []

t_idx = np.arange(1, num_epochs+1)

# raw data with different learning rates
fn_1 = './data/results/mlp/config_raw_1.json'

with open(fn_1, 'r') as fh:
    tmp = json.load(fh)
    config_1 = tmp['config']
    configs.append(config_1)
    learning_curve_1 = tmp['learning_curve']
    learning_curves.append(learning_curve_1)

fn_2 = './data/results/mlp/config_raw_2.json'

with open(fn_2, 'r') as fh:
    tmp = json.load(fh)
    config_2 = tmp['config']
    configs.append(config_2)
    learning_curve_2 = tmp['learning_curve']
    learning_curves.append(learning_curve_2)
    
fn_3 = './data/results/mlp/config_raw_3.json'

with open(fn_3, 'r') as fh:
    tmp = json.load(fh)
    config_3 = tmp['config']
    configs.append(config_3)
    learning_curve_3 = tmp['learning_curve']
    learning_curves.append(learning_curve_3)

fn_4 = './data/results/mlp/config_raw_8.json'

with open(fn_4, 'r') as fh:
    tmp = json.load(fh)
    config_4 = tmp['config']
    configs.append(config_4)
    learning_curve_4 = tmp['learning_curve']
    learning_curves.append(learning_curve_4)

fn_5 = './data/results/mlp/config_raw_9.json'

with open(fn_5, 'r') as fh:
    tmp = json.load(fh)
    config_5 = tmp['config']
    configs.append(config_5)
    learning_curve_5 = tmp['learning_curve']
    learning_curves.append(learning_curve_5)
    
fn_6 = './data/results/mlp/config_raw_10.json'

with open(fn_6, 'r') as fh:
    tmp = json.load(fh)
    config_6 = tmp['config']
    configs.append(config_6)
    learning_curve_6 = tmp['learning_curve']
    learning_curves.append(learning_curve_6)
        
fig, ax = plt.subplots()
ax.plot(t_idx, learning_curve_1, 'r', label='lr = 0.01')
ax.plot(t_idx, learning_curve_2, 'g', label='lr = 0.005')
ax.plot(t_idx, learning_curve_3, 'b', label='lr = 0.02')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FF00')
plt.title("Learning curves of raw datasets at different learning rates")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")                
plt.show()


# preprocessed data with different learning rates

fn_1 = './data/results/mlp/config_pre_1.json'

with open(fn_1, 'r') as fh:
    tmp = json.load(fh)
    config_1 = tmp['config']
    configs.append(config_1)
    learning_curve_1 = tmp['learning_curve']
    learning_curves.append(learning_curve_1)

fn_2 = './data/results/mlp/config_pre_2.json'

with open(fn_2, 'r') as fh:
    tmp = json.load(fh)
    config_2 = tmp['config']
    configs.append(config_2)
    learning_curve_2 = tmp['learning_curve']
    learning_curves.append(learning_curve_2)
    
fn_3 = './data/results/mlp/config_pre_3.json'

with open(fn_3, 'r') as fh:
    tmp = json.load(fh)
    config_3 = tmp['config']
    configs.append(config_3)
    learning_curve_3 = tmp['learning_curve']
    learning_curves.append(learning_curve_3)
        
fig, ax = plt.subplots()
ax.plot(t_idx, learning_curve_1, 'r', label='lr = 0.01')
ax.plot(t_idx, learning_curve_2, 'g', label='lr = 0.005')
ax.plot(t_idx, learning_curve_3, 'b', label='lr = 0.02')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FF00')
plt.title("Learning curves of preprocessed datasets at different learning rates")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")                
plt.show()


# raw data with different drop rate

fn_1 = './data/results/mlp/config_raw_1.json'

with open(fn_1, 'r') as fh:
    tmp = json.load(fh)
    config_1 = tmp['config']
    configs.append(config_1)
    learning_curve_1 = tmp['learning_curve']
    learning_curves.append(learning_curve_1)

fn_2 = './data/results/mlp/config_raw_4.json'

with open(fn_2, 'r') as fh:
    tmp = json.load(fh)
    config_2 = tmp['config']
    configs.append(config_2)
    learning_curve_2 = tmp['learning_curve']
    learning_curves.append(learning_curve_2)
    
fn_3 = './data/results/mlp/config_raw_5.json'

with open(fn_3, 'r') as fh:
    tmp = json.load(fh)
    config_3 = tmp['config']
    configs.append(config_3)
    learning_curve_3 = tmp['learning_curve']
    learning_curves.append(learning_curve_3)
        
fig, ax = plt.subplots()
ax.plot(t_idx, learning_curve_1, 'r', label='dr = 0.2')
ax.plot(t_idx, learning_curve_2, 'g', label='dr = 0.4')
ax.plot(t_idx, learning_curve_3, 'b', label='dr = 0.1')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FF00')
plt.title("Learning curves of raw datasets at different drop rates")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")                
plt.show()

# preprocessed data with different drop rate

fn_1 = './data/results/mlp/config_pre_1.json'

with open(fn_1, 'r') as fh:
    tmp = json.load(fh)
    config_1 = tmp['config']
    configs.append(config_1)
    learning_curve_1 = tmp['learning_curve']
    learning_curves.append(learning_curve_1)

fn_2 = './data/results/mlp/config_pre_4.json'

with open(fn_2, 'r') as fh:
    tmp = json.load(fh)
    config_2 = tmp['config']
    configs.append(config_2)
    learning_curve_2 = tmp['learning_curve']
    learning_curves.append(learning_curve_2)
    
fn_3 = './data/results/mlp/config_pre_5.json'

with open(fn_3, 'r') as fh:
    tmp = json.load(fh)
    config_3 = tmp['config']
    configs.append(config_3)
    learning_curve_3 = tmp['learning_curve']
    learning_curves.append(learning_curve_3)
        
fig, ax = plt.subplots()
ax.plot(t_idx, learning_curve_1, 'r', label='dr = 0.2')
ax.plot(t_idx, learning_curve_2, 'g', label='dr = 0.4')
ax.plot(t_idx, learning_curve_3, 'b', label='dr = 0.1')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FF00')
plt.title("Learning curves of preprocessed datasets at different drop rates")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")                
plt.show()

#raw data with different expotential lr

fn_1 = './data/results/mlp/config_raw_1.json'

with open(fn_1, 'r') as fh:
    tmp = json.load(fh)
    config_1 = tmp['config']
    configs.append(config_1)
    learning_curve_1 = tmp['learning_curve']
    learning_curves.append(learning_curve_1)

fn_2 = './data/results/mlp/config_raw_6.json'

with open(fn_2, 'r') as fh:
    tmp = json.load(fh)
    config_2 = tmp['config']
    configs.append(config_2)
    learning_curve_2 = tmp['learning_curve']
    learning_curves.append(learning_curve_2)
    
fn_3 = './data/results/mlp/config_raw_7.json'

with open(fn_3, 'r') as fh:
    tmp = json.load(fh)
    config_3 = tmp['config']
    configs.append(config_3)
    learning_curve_3 = tmp['learning_curve']
    learning_curves.append(learning_curve_3)
        
fig, ax = plt.subplots()
ax.plot(t_idx, learning_curve_1, 'r', label='no exp lr')
ax.plot(t_idx, learning_curve_2, 'g', label='exp lr = 0.95')
ax.plot(t_idx, learning_curve_3, 'b', label='exp lr = 0.90')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FF00')
plt.title("Learning curves of raw datasets at different drop rates")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")                
plt.show()

#preprocessed data with different expotential lr

fn_1 = './data/results/mlp/config_pre_1.json'

with open(fn_1, 'r') as fh:
    tmp = json.load(fh)
    config_1 = tmp['config']
    configs.append(config_1)
    learning_curve_1 = tmp['learning_curve']
    learning_curves.append(learning_curve_1)

fn_2 = './data/results/mlp/config_pre_6.json'

with open(fn_2, 'r') as fh:
    tmp = json.load(fh)
    config_2 = tmp['config']
    configs.append(config_2)
    learning_curve_2 = tmp['learning_curve']
    learning_curves.append(learning_curve_2)
    
fn_3 = './data/results/mlp/config_pre_7.json'

with open(fn_3, 'r') as fh:
    tmp = json.load(fh)
    config_3 = tmp['config']
    configs.append(config_3)
    learning_curve_3 = tmp['learning_curve']
    learning_curves.append(learning_curve_3)
        
fig, ax = plt.subplots()
ax.plot(t_idx, learning_curve_1, 'r', label='no exp lr')
ax.plot(t_idx, learning_curve_2, 'g', label='exp lr = 0.95')
ax.plot(t_idx, learning_curve_3, 'b', label='exp lr = 0.90')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FF00')
plt.title("Learning curves of preprocessed datasets at different drop rates")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")                
plt.show()