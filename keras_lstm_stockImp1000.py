'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
https://github.com/fchollet/keras/blob/master/examples/stateful_lstm.py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (25, 1, 50)               10400
_________________________________________________________________
lstm_2 (LSTM)                (25, 50)                  20200
_________________________________________________________________
dense_1 (Dense)              (25, 1)                   51
=================================================================
Total params: 30,651
Trainable params: 30,651
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 25
Epoch 1/1
50000/50000 [==============================] - 14s - loss: 170.6959
Epoch 24 / 25
Epoch 1/1
50000/50000 [==============================] - 15s - loss: 3.9509
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam,Adamax
import pandas as pd

# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 63 #25
epochs = 3001
# number of elements ahead that are used to make the prediction
lahead = 63


def gen_cosine_amp(amp=100, period=2000, x0=0, xn=253, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing

    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    data = pd.read_csv('data_5m.csv')
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        #cos[i, 0, 0] = 0.5*amp * (np.cos(2 * np.pi * idx / period))
        #cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)+100*np.random.uniform(-1.0, +1.0)
        print(data[i])
        #cos[i, 0, 0] = data
    return cos
# データの変換方法
def convert_data(df):
    # 元データを破壊したくないので一旦コピー
    df = df.copy() 
    # np.array型で返す
    return np.array(df)


# データをロード
#from sklearn import model_selection
#from sklearn.model_selection import train_test_split
data = pd.read_csv('data_5m.csv')
data = convert_data(data)

cos = np.zeros(((len(data) - 0) * 1, 1, 1))
print('data[1]',data)
print('Generating Data...')
for i in range(len(cos)):
    cos[i,0,0] = 5000*(data[i]-data[0])/data[np.argmax(data)]    #gen_cosine_amp()

print('Input shape:', cos.shape)
print(cos)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = cos[i]  #np.mean(cos[i + 1:i + 1 + 1])  #lahead =1
expected_output1 = np.zeros((len(cos), 1))
for i in range(len(cos)):
    expected_output1[i, 0] = cos[i]  #np.mean(cos[i + 1:i + 1 + 1])  #lahead =1

print('Output shape:', expected_output.shape)
lr=0.0001
print('Creating Model...')
model = Sequential()
model.add(LSTM(100,
               input_shape=(tsteps, 1),
               batch_size=batch_size,
               return_sequences=False,
               stateful=True))
#model.add(Dropout(0.25))
"""
model.add(LSTM(100,
               return_sequences=False,
               stateful=True))
#model.add(Dropout(0.5))
"""
model.add(Dense(1))
opt=Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.)
model.compile(loss='mse', optimizer=opt)   #'rmsprop')
model.summary()
#model.load_weights('params_model_lstm_epoch_10000.hdf5')

print('Training')
for i in range(0,epochs):
    #plt.figure(num=None, figsize=(15, 15), dpi=60)
    print('Epoch', i, '/', epochs)

    # Note that the last state for sample i in a batch will
    # be used as initial state for sample i in the next batch.
    # Thus we are simultaneously training on batch_size series with
    # lower resolution than the original series contained in cos.
    # Each of these series are offset by one step and can be
    # extracted with cos[i::batch_size].

    model.fit(cos, expected_output,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              shuffle=False)
    model.reset_states()
    
    #print('Predicting')
    #predicted_output = model.predict(cos, batch_size=batch_size)
    if i%100==0:
        # save weights every epoch
        model.save_weights(
              'params_model_lstm_epoch_{0:03d}.hdf5'.format(i), True)
        print('Predicting')
        predicted_output = model.predict(cos, batch_size=batch_size)
        #lr=0.5*lr
        #opt=Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.)
        #model.compile(loss='mse', optimizer=opt) 
        plt.figure(num=None, figsize=(15, 15), dpi=60)
        print('Plotting Results')
        plt.subplot(3, 1, 1)
        plt.plot(expected_output)
        plt.ylim(-25, 25)
        plt.xlim(0, 260)
        plt.subplot(3, 1, 2)
        plt.plot(expected_output1)
        plt.ylim(-25, 25)
        plt.xlim(0, 260)

        plt.title('Expected')
        plt.subplot(3, 1, 3)
        plt.plot(predicted_output)
        plt.ylim(-25, 25)
        plt.xlim(0, 260)
        plt.title('Predicted')
        plt.pause(3)
        plt.savefig('plot_epoch_{0:03d}_lstm.png'.format(i), dpi=60)
        plt.close()
    else:
        continue
        
