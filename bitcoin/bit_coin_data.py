import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

ROW_NUM = 1300 #Cut off row between training data and validation data
TEST_DATA_SIZE = 10 #Number of days tested

data_set=pd.read_csv('BTC-USD.csv')      #reading csv file
data_set=data_set.iloc[:,1:2]        #selecting the second column that contains starting price info
data_set=data_set.values            #converting to 2d array

ytest=data_set[ROW_NUM+1:ROW_NUM+TEST_DATA_SIZE+1] #Extract the validation data

#Scaling the data
sc = MinMaxScaler()                          #scaling using normalisation 
data_set = sc.fit_transform(data_set)
xtrain=data_set[0:ROW_NUM]                   #input values of rows [0-ROW_NUM]           
ytrain=data_set[1:ROW_NUM+1]                 #input values of rows [1-ROW_NUM+1]
xtest=data_set[ROW_NUM:ROW_NUM+TEST_DATA_SIZE]

xtrain = np.reshape(xtrain, (ROW_NUM, 1, 1)) #Reshaping into required shape for Keras

regressor=Sequential()                                                      #initialize the RNN
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))      #adding input layerand the LSTM layer 
regressor.add(Dense(units=1))                                               #adding output layers
regressor.compile(optimizer='adam',loss='mean_squared_error')               #compiling the RNN
regressor.fit(xtrain,ytrain,batch_size=64,epochs=1000)                      #fitting the RNN to the training set  

#getting the predicted BTC value of the week of 11/14 2017            
xtest = np.reshape(xtest, (TEST_DATA_SIZE, 1, 1))
predicted_price = regressor.predict(xtest)
predicted_price = sc.inverse_transform(predicted_price)

plt.plot(ytest, color = 'red', label = 'Real BTC Value')
plt.plot(predicted_price, color = 'blue', label = 'Predicted BTC Value')
plt.title('BTC Value Prediction')
plt.xlabel('Days')
plt.ylabel('BTC Value')
plt.legend()
plt.show()