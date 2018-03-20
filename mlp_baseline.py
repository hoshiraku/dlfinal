#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:55:40 2018

@author: hoshiraku
"""

import numpy as np
from sklearn.model_selection import KFold
from data import load_data_raw
from data import load_data_standardization
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



if __name__ == '__main__':
    # many baselines
    
    # Random Forest Regression
    from sklearn.ensemble import RandomForestRegressor
    # load X, y
    inputs_raw, labels_raw = load_data_raw()
    inputs_pre, labels_pre = load_data_standardization()
    #print(inputs)
    #labels = labels.ravel()
    #print(labels)
    #reg_ranfor = RandomForestRegressor(max_depth=2, random_state=0)
    #reg_ranfor.fit(inputs, labels)
    
    # cross validation for MLP model
    
    kf = KFold(n_splits=3, shuffle=True)
    
    print("Random Forest model:")
    avg_error_ranfor = 0.0
    for train, val in kf.split(labels_raw):
        X_train, y_train, X_val, y_val = inputs_raw[train], labels_raw[train], inputs_raw[val], labels_raw[val]
        #construct a random forest regression model
        reg_ranfor = RandomForestRegressor(max_depth=3, random_state=0)
        #convert column vector to 1-d array
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        #train the model
        reg_ranfor.fit(X_train, y_train)
        y_pred = reg_ranfor.predict(X_val)
        #print(y_pred)
        error_ranfor = mean_squared_error(y_val, y_pred)
        avg_error_ranfor += error_ranfor * len(y_val) / len(labels_raw)
        print("one cross validation MSE: %.3f" % error_ranfor)
    
    print("3-fold cross validation average loss: %.3f\n" %avg_error_ranfor)
    
    # Decision Tree Regression
    print("Decision Tree model:")
    avg_error_detr = 0.0
    for train, val in kf.split(labels_raw):
        X_train, y_train, X_val, y_val = inputs_raw[train], labels_raw[train], inputs_raw[val], labels_raw[val]
        #construct a decision tress regression model
        reg_detr = DecisionTreeRegressor(max_depth=2)
        #train the model
        reg_detr.fit(X_train, y_train)
        y_pred = reg_detr.predict(X_val)
        error_detr = mean_squared_error(y_val, y_pred)
        avg_error_detr += error_detr * len(y_val) / len(labels_raw)
        print("one cross validation MSE: %.3f" % error_detr)
    
    print("3-fold cross validation average loss: %.3f" %avg_error_detr)
    
    # Linear Regression
    print("Linear model:")
    avg_error_lin = 0.0
    for train, val in kf.split(labels_raw):
        X_train, y_train, X_val, y_val = inputs_raw[train], labels_raw[train], inputs_raw[val], labels_raw[val]
        #construct a decision tress regression model
        reg_lin = LinearRegression()
        #train the model
        reg_lin.fit(X_train, y_train)
        y_pred = reg_lin.predict(X_val)
        error_lin = mean_squared_error(y_val, y_pred)
        avg_error_lin += error_lin * len(y_val) / len(labels_raw)
        print("one cross validation MSE: %.3f" % error_lin)
    
    print("3-fold cross validation average loss: %.3f" %avg_error_lin)
    