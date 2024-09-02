# from statsmodels.nonparametric.kernel_regression import KernelReg
# import pandas as pd
# import pandas_datareader as pdr
# import matplotlib.pyplot as plt
# import investpy
# from scipy.signal import argrelextrema
# import numpy as np
# from collections import defaultdict
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
# from scipy.signal import argrelextrema
# from statsmodels.nonparametric.kernel_regression import KernelReg
def find_MaxtimeStep(df):
    x = df.loc[df['TREND'].notnull()].index.tolist()
    max_distance =0 
    for i in range (0,len(x)-1):
        if x[i+1]-x[i] >max_distance:
            max_distance = x[i+1]-x[i]
    return max_distance
def get_fixedSizeCandle(candles,md):
    candles_array = np.array(candles)
    Fixed_MaxSizeCandles = np.zeros((md,1),dtype=float)
    Fixed_MaxSizeCandles[md-candles_array.shape[0]:,:1] =candles_array
    return Fixed_MaxSizeCandles
def PrepareData( df):
    Max_TimeStep = find_MaxtimeStep(df)
    x=list()
    y=list()
    i=0
    candles = list()

    while i<len(df):
        candle =list()
        candle = np.zeros(1)
        if not(pd.isnull(df.at[i,'TREND'])):
            y.append(df.at[i,'TREND'])
            if len(candles)>0:
                    x.append(get_fixedSizeCandle(candles,Max_TimeStep))
            candles= list()
        else:
            candle[0]= df.iloc[i,5] -df.iloc[i-1,5]
            candles.append(candle)
        i = i+1
    x.append(get_fixedSizeCandle(candles,Max_TimeStep))
    return np.array(x),np.array(y)

excel_data = pd.read_excel('Full_Sample.xlsx')
df = pd.DataFrame(excel_data, columns=['DATE', 'TIME', 'OPEN','HIGH','LOW','CLOSE','TREND'])
x,y = PrepareData(df)

scaler = MinMaxScaler()

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
X_Train = x[:30,:,:]
X_Test = x[30:,:,:]
X_Train = scaler.fit_transform(X_Train.reshape(-1, X_Train.shape[-1])).reshape(X_Train.shape)
X_Test = scaler.transform(X_Test.reshape(-1, X_Test.shape[-1])).reshape(X_Test.shape)

Y_Train = onehot_encoded[:30,:]
Y_Test = onehot_encoded[30:,:]

model = Sequential()
model.add(LSTM(4, input_shape=(100, 1)))

model.add(Dense(3,activation= 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
history=model.fit(X_Train, Y_Train, epochs=100,  verbose=2)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

testPredict = model.predict(X_Test)
loss = model.evaluate(X_Test,Y_Test,verbose=2)
print(loss)
