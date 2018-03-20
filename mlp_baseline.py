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
    avg_score_ranfor = 0.0
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
        score_ranfor = reg_ranfor.score(X_val, y_val)
        avg_score_ranfor += score_ranfor * len(y_val) / len(labels_raw)
        
        error_ranfor = mean_squared_error(y_val, y_pred)
        avg_error_ranfor += error_ranfor * len(y_val) / len(labels_raw)
        
        print("one cross validation MSE: % f" % error_ranfor)
        print("one cross validation score: % f" % score_ranfor)
    
    print("3-fold cross validation MSE: % f" %avg_error_ranfor)
    print("3-fold cross validation score: % f" %avg_score_ranfor)
    
    # Decision Tree Regression
    print("Decision Tree model:")
    avg_error_detr = 0.0
    avg_score_detr = 0.0
    
    for train, val in kf.split(labels_raw):
        X_train, y_train, X_val, y_val = inputs_raw[train], labels_raw[train], inputs_raw[val], labels_raw[val]
        #construct a decision tress regression model
        reg_detr = DecisionTreeRegressor(max_depth=2)
        #train the model
        reg_detr.fit(X_train, y_train)
        y_pred = reg_detr.predict(X_val)
        error_detr = mean_squared_error(y_val, y_pred)
        score_detr = reg_detr.score(X_val, y_val)
        
        avg_error_detr += error_detr * len(y_val) / len(labels_raw)
        avg_score_detr += score_detr * len(y_val) / len(labels_raw)
        
        print("one cross validation MSE: % f" % error_detr)
        print("one cross validation score: % f" % score_detr)
    
    print("3-fold cross validation MSE: % f" %avg_error_detr)
    print("3-fold cross validation score: % f" %avg_score_detr)
    
    # Linear Regression
    print("Linear model:")
    avg_error_lin = 0.0
    avg_score_lin = 0.0
    
    for train, val in kf.split(labels_raw):
        X_train, y_train, X_val, y_val = inputs_raw[train], labels_raw[train], inputs_raw[val], labels_raw[val]
        #construct a decision tress regression model
        reg_lin = LinearRegression()
        #train the model
        reg_lin.fit(X_train, y_train)
        y_pred = reg_lin.predict(X_val)
        score_lin = reg_lin.score(X_val, y_val)
        error_lin = mean_squared_error(y_val, y_pred)
        
        avg_score_lin += score_lin * len(y_val) / len(labels_raw)
        avg_error_lin += error_lin * len(y_val) / len(labels_raw)
        
        print("one cross validation MSE: % f" % error_lin)
        print("one cross validation score: % f" % score_lin)
    
    print("3-fold cross validation MSE: % f" %avg_error_lin)
    print("3-fold cross validation score: % f" %avg_score_lin)