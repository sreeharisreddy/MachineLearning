# Loading the library with the iris dataset
from sklearn.datasets import load_iris
# Loading scikit's randomforest classifier library
from sklearn.ensemble import RandomForestClassifier
#loading pandas
import pandas as pd
#loading numpy
import numpy as np
#seeting random seed
np.random.seed(0)
#creating an object called iris with the iris data
iris = load_iris()
#print(iris)
#craeting a data frame with foure featur variables
df = pd.DataFrame(iris.data,columns=iris.feature_names)
#df.head()
#Adding a new column for the species name
df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)
#Creating the Test and Train data randomly
df['is_train'] = np.random.uniform(0,1,len(df)) <= .75
df.head()
#Creating adataframe with test rows and training rows
train,test = df[df['is_train']==True], df[df['is_train']==False]
#Show the number of observations for the test and training dataframes
print('Numbe of observations in the training dataset:',len(train))
print('Numbe of observations in the test dataset:',len(test))
#Creating the list of the feature column's names
features = df.columns[:4]
print(features)
#Converting each species into digits
y=pd.factorize(train['species'])[0]
#Creating Random ForestClassifier 
clf = RandomForestClassifier(n_jobs=2,random_state=0)
#Training the classifier
clf.fit(train[features],y)
#Applying the trained classifier to the test
clf.predict(test[features])
#Viewing  the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]
#mapping names for thr plans for each plant class
preds = iris.target_names[clf.predict(test[features])]
#view thr predicted species for the first five observations
preds[0:5]
#view the Actual species for the five first observation
test['species'].head()
#creating confusion matrix
pd.crosstab(test['species'],preds,rownames=['Actual Species'],colnames=['Predicted Species'])
#Predicting for new dataset
preds = iris.target_names[clf.predict( [[9.0,3.6,7.4,7.0],[5.0,3.6,1.4,2.0]])]
preds