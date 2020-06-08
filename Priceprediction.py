import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#Get the stock quote
df = web.DataReader('AAPL',data_source='yahoo',start='2012-01-01',end='2020-06-07')
#df.shape
#visualize the closing price history
plt.figure(figsize=(16,8))
plt.title("Close Price History")
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()
data = df.filter(['Close'])
#convertthe data from to numpy array
dataset = data.values
#Get the number of rows to train the model
training_data_len = math.ceil(len(dataset) * .8)
#training_data_len
#sclaing the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
#creating the training dataset
#create the scaled training dataset
train_data = scaled_data[0:training_data_len , :]
# split the data into x_train and y_train data sets
x_train = [] #indipendent variables
y_train = [] #target variables

for i in range (60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i, 0])
    if i <= 61 :
        print (x_train)
        print (y_train)
        print()
		
#convert the x_trzin and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#Re shape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape
#Build the LSTM Model
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50,return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

#compile the Model
model.compile(optimizer='adam',loss='mean_squared_error')
#Train the model
model.fit(x_train, y_train,batch_size=1, epochs=1)
#Batch size Total no of training example in a single batch 
#epochs is the no of iterations when a pass over through the entire dataset

#create the testing data set
#create a new array containing scaled values from index 1543 ,to 2003
test_data = scaled_data[training_data_len - 60: , :]                     
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
#vonvert the data to numpy array(
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #reversing unsclaing

#Get the root mean squared error (RMSE) , is a good messure of how accurate the model predicts the response,
#it is the standard deviations of the reseduals , the lower values means better fit of the model
rmse = np.sqrt(np.mean(predictions-y_test)**2)
rmse
#plot the data
train = data[:training_data_len] 
valid = data[training_data_len:]
valid['Predictions'] = predictions
#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()
#Get the quote
apple_quote = web.DataReader('AAPL',data_source='yahoo', start='2012-01-01',end='2020-06-07')
#create a new data frame
new_df = apple_quote.filter(['Close'])
#Get the last 60days Close price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_Scaled = scaler.transform(last_60_days)
#create an empty list 
X_test = []
#Append the past 60 days
X_test.append(last_60_days_Scaled)
#Conver the x_test to numpy array
X_test = np.array(X_test)
#Reshape the data 
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
#COmpare the predicted price with latest price
apple_quote2 = web.DataReader('AAPL',data_source='yahoo', start='2020-06-08',end='2020-06-08')
print(apple_quote2['Close'])
