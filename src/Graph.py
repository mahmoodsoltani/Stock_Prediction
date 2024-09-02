# from statsmodels.nonparametric.kernel_regression import KernelReg
# import pandas as pd
# import pandas_datareader as pdr
# import matplotlib.pyplot as plt
# import investpy
# from scipy.signal import argrelextrema
# import numpy as np
# from collections import defaultdict
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Attention
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
    Fixed_MaxSizeCandles = np.zeros((md,candles_array.shape[1]),dtype=float)
    Fixed_MaxSizeCandles[md-candles_array.shape[0]:,:candles_array.shape[1]] =candles_array
    return Fixed_MaxSizeCandles
def PrepareData( df,Max_TimeStep):
    
    x=list()
    y=list()
    i=0
    candles = list()

    while i<len(df):
        candle =list()
        candle = np.zeros(4)
        if not(pd.isnull(df.at[i,'TREND'])):
            y.append(df.at[i,'TREND'])
            if len(candles)>0:
                    x.append(get_fixedSizeCandle(candles,Max_TimeStep))
            candles = list()
        else:
            for j in range (0,4):
                candle[j] = df.iloc[i,j+2] -df.iloc[i-1,j+2]
                #candle[j] = df.iloc[i,j+2]
        candles.append(candle)
        i = i+1
    x.append(get_fixedSizeCandle(candles,Max_TimeStep))
    return np.array(x),np.array(y)

def Create_Model(num_timesteps,num_features):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(num_timesteps, num_features)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(16))
    model.add(Dense(3,activation= 'softmax'))
    return model
def Show_history_plot(history):
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

excel_data = pd.read_excel('Full_Sample.xlsx')
df = pd.DataFrame(excel_data, columns=['DATE', 'TIME', 'OPEN','HIGH','LOW','CLOSE','TREND'])
Max_TimeStep = find_MaxtimeStep(df)
x,y = PrepareData(df,Max_TimeStep)
scaler = StandardScaler()
for i in range(0,x.shape[0]):
    x[i]= scaler.fit_transform(x[i])
#x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
train_Persent = 15
X_Train = x[:int((100-train_Persent)*x.shape[0]/100),:,:]
X_Test = x[(X_Train.shape[0]):,:,:]


Y_Train = onehot_encoded[:X_Train.shape[0],:]
Y_Test = onehot_encoded[X_Train.shape[0]:,:]

model = Create_Model(Max_TimeStep,4)
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
history= model.fit(X_Train, Y_Train, epochs=100,  verbose=0)
#Show_history_plot(history)
testPredict = model.predict(X_Test)
print(model.evaluate(X_Test,Y_Test,verbose=0))
predicted = list()
classes = ['Uptrend','Downtrend','sidetrend']
for row in predicted:
    
print(predicted)