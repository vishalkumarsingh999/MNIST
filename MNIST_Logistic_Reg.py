# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 22:00:53 2021

@author: Vishal Kumar Singh
"""
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
 

data=pd.read_csv("MNIST_Dataset/train.csv")
y=data.iloc[:,0]
x=data.iloc[:,1:]


from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x,y,random_state=False,test_size=0.3)

from sklearn.linear_model import LogisticRegression
lin=LogisticRegression()
lin.fit(x_tr,y_tr)



y_pred=lin.predict(x_te)

from sklearn.metrics import confusion_matrix

con=confusion_matrix(y_te,y_pred)
total=0;
correct=0;
for i in range(10):
    for j in range(10):
        total+=con[i][j];
        if(i==j):
            correct+=con[i][j];

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_te,y_pred)
print("accuracy=",(correct*100)/total," %\naccuracy=",acc*100)
