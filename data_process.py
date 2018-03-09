#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:03:43 2018

@author: hoshiraku
"""

import os
import sys
import glob
import json

import numpy as np

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

N = len(configs) #265
#print(N)
n_epochs = len(learning_curves[0]) #40
#print(n_epochs)

#print(learning_curves[0])
#print(configs[0])

#print(configs[0]['batch_size'])

X_datasets = []
y_labelsets = []

for index_i in range(N): # 265 configures and learning curves

    #extract hyperparameters from configurations' data
    batch_size = configs[index_i]['batch_size']
    log2_n_units_2 = configs[index_i]['log2_n_units_2']
    log10_learning_rate = configs[index_i]['log10_learning_rate']
    log2_n_units_3 = configs[index_i]['log2_n_units_3']
    log2_n_units_1 = configs[index_i]['log2_n_units_1']

    #the i-th configs
    learning_curve_i =  learning_curves[index_i];   
        
    for index_j in range(n_epochs - 4): # 36
        
        parameter_in = [];

        #combine the parameters to a row of X datasets
        parameter_in.append(batch_size)
        parameter_in.append(log2_n_units_2)
        parameter_in.append(log10_learning_rate)
        parameter_in.append(log2_n_units_3)
        parameter_in.append(log2_n_units_1)
        #use the last 4 point to predict the next 1 point
        parameter_in.append(learning_curve_i[index_j])
        parameter_in.append(learning_curve_i[index_j+1])
        parameter_in.append(learning_curve_i[index_j+2])
        parameter_in.append(learning_curve_i[index_j+3])
        
        X_datasets.append(parameter_in)
        
        #add the labels to the row of y labelsets
        y_labelsets.append(learning_curve_i[index_j+4])

print(y_labelsets)
print(len(y_labelsets))
        
        