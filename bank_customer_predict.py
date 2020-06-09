import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
print(tf.__version__)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #Used for converting labels to numbers(0 or 1)
dataset = pd.read_csv('CustomerModelling.csv')
dataset.head()
x = dataset.drop(labels=['RowNumber','CustomerId','Surname','Exited'], axis=1)
y = dataset['Exited']
x.head()
#y.head()
label1 = LabelEncoder()
x['Geography'] = label1.fit_transform(x['Geography'])
x.head(20)
label2 = LabelEncoder()
x['Gender'] = label2.fit_transform(x['Gender'])
x.head(10)
#Since Gender and Geography are not have any waightage based on the number ,these are catagorical valyes .(0.1,2,3..all ahave same).
#We need to convert into one hot encoding
x = pd.get_dummies(x,drop_first=True,columns=['Geography'])
###future standardization, since some of the values are large numbers we need to scale
from sklearn.preprocessing import StandardScaler
x_train , x_test ,y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 0,stratify = y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
y_train
model = Sequential()
model.add(Dense(x.shape[1],activation='relu',input_dim=x.shape[1])) #First input layer
model.add(Dense(128,activation='relu')) #Hidden layer with 128 newrons and no input_dim required since it is not a inputlayer
model.add(Dense(1,activation='sigmoid')) #Outplut layer , here ther is 2 only 2 values (1 or 0) hence we required only single node ,1 means customer with tge bank and o means customer going to leave the bank.
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #adam optimizer is the stocastic gradiant optimizer , we are not used batch radiant gradiant descent alagorithm which take a lot of time to train the model
#loss function we have  binary_crossentropy  , since we have to predict binary classificatin other wise we have to use other
#Start fitting the model
model.fit(x_train,y_train.to_numpy(),batch_size=10,epochs = 10 , verbose = 1) 
#since x_train is numpy array , the y_train alos should be numpy array , batchsize means in stocastic gradiants how many rows or datapoints 
#going to take caliculate the new weights . If batch size=1 that means every input it will update the weights that becomes toomuch random ,
#and if batchsize = total size of data ,then the stocastica gradiant becomes a kind of batch gradiant .
#epochs = 10   we can try for 20 also , we need to see the acurracy , if the accurracy not much changed we should not increase the epoch
#verbose means verbose mode give some logs and give progress bar
y_pred = model.predict_classes(x_test)
#check the prediction and actual manually
y_pred
y_test
model.evaluate(x_test,y_test.to_numpy()) #evoluating the accuracy using tensorflow
from sklearn.metrics import confusion_matrix, accuracy_score  #Evoluating the accurracy with sklearn metrics methods
confusion_matrix(y_test,y_pred) # shows howmany 0s are correct and incorrect and how many ones are correct and incorrect
accuracy_score(y_test,y_pred)   #Accurracy is same bwter tensorflow methos and sklearn