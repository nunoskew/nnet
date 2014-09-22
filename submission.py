# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:04:14 2014

@author: sku
"""
from pre_process import *
from nnet import *   
import numpy as np
import pandas as pd

def submit(filename):
    """(string) -> None
    
    Runs perceptron on the titanic dataset and generates
    a submission file 'filename'.csv
    
    """
    
    #read data
    train,test=read_data()
    #does some of the trevor stephens pre-processing, check docstring of function
    train,test,y_matrix,id_matrix=pre_process_(train,test)
    #turns dataframes into matrices so the logistic regression code doesnt flip out
    train_,test_=turn_into_matrices(train,test)
    #set nnet params
    layer_list=[10]
    epsilon_init=0.12
    alpha=0.1
    n_iter=1000
    lambd=0
    threshold=0.5
    seed=1234
    #get theta
    theta=nnet(train_,y_matrix,layer_list,alpha,epsilon_init,n_iter,lambd,seed)
    #get predictions in the test set
    pred=predict_nnet(test_,y_matrix,threshold,layer_list,theta,epsilon_init,seed)
    #merge PassengerId and predictions to create the submission DataFrame
    df=pd.DataFrame(np.concatenate([id_matrix,pred],axis=1))
    #set up column names
    df.columns=['PassengerId','Survived']
    #turn PassengerId column from float to int
    df['PassengerId']=df['PassengerId'].astype('int')
    #turn predictions from float into binary int 0,1
    df['Survived']=(df['Survived']>threshold)*1
    #create submission csv
    df.to_csv(filename,header=True,index=False)