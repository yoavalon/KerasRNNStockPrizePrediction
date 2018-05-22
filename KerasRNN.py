import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


link = 'https://raw.githubusercontent.com/llSourcell/How-to-Predict-Stock-Prices-Easily-Demo/master/sp500.csv'    
df = pd.read_csv(link)    
training_set = df.iloc[0:1000,0].values.reshape(-1, 1)
test_set = df.iloc[1000:1400,0].values.reshape(-1, 1)

sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

#print(training_set_scaled)

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(60, 1000) :
  X_train.append(training_set_scaled[i-60:i,0])
  y_train.append(training_set_scaled[i,0])
  
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

for i in range(60, 400) :
  X_test.append(test_set_scaled[i-60:i,0])
  y_test.append(test_set_scaled[i,0])
  
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences = True, input_shape = (X_train.shape[1],1)) )
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs=10, batch_size=32)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(test_set, color='red', label='Real')
plt.plot(predicted_stock_price, color='blue', label='Predicted')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

plt.plot(training_set, color='red', label='Training')
plt.title('Training Stock Price ')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


