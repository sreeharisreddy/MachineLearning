#The bank can use this model to decide whether it should approve loan request from a perticular customer are not
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#Loading data from a file
fullFileName = 'E:/MachineLearning/MachineLearning/DecisionTreeData.txt'
balance_data=pd.read_csv(fullFileName,sep=',',header=0)
balance_data.shape
balance_data.head()

#Seperating the traget variable
X=balance_data.values[:,1:5] #Data to train 
Y=balance_data.values[:,0] #Answers for training
#Split data set into test and train
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.3,random_state=100)
#Function to perform training with entropy
clf_entropy = DecisionTreeClassifier(criterion = 'entropy' , random_state=100,max_depth=3,min_samples_leaf=5)
clf_entropy.fit(x_train,y_train)
#Function to make predictions
y_pred_en=clf_entropy.predict(x_test)
y_pred_en
#Checking the accuracy
print("Accuracy is :",accuracy_score(y_test,y_pred_en)*100)