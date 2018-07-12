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
from keras.layers import Dense, LSTM
from keras.optimizers import Adam


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 25
epochs = 301
# number of elements ahead that are used to make the prediction
lahead = 2000


def gen_cosine_amp(amp=80, period=200, x0=0, xn=5000, step=1, k=0.0001):
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
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        #cos[i, 0, 0] = amp * (np.cos(2 * np.pi * idx / period)* np.exp(-0.0005 * idx)+0.3*np.sin(2 * np.pi * idx / period/1.53)+0.5*np.sin(2 * np.pi * idx / period/9.53)*np.random.uniform(-1.0, +1.0))
        cos[i, 0, 0] = amp * (np.cos(2 * np.pi * idx / period))
        cos[i, 0, 0] = cos[i, 0, 0]* np.exp(-k * idx)
    return cos

print('Generating Data...')
cos = gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = cos[i]  #np.mean(cos[i + 1:i + 1 + 1])  #lahead =1
expected_output1 = np.zeros((len(cos), 1))
output1 = np.zeros((len(cos), 1))
for i in range(len(cos)):
    expected_output1[i, 0] = cos[i]  #np.mean(cos[i + 1:i + 1 + 1])  #lahead =1    

print('Output shape:', expected_output.shape)

print('Creating Model...')
model = Sequential()
model.add(LSTM(100,
               input_shape=(tsteps, 1),
               batch_size=batch_size,
               return_sequences=False,
               stateful=True,
              dropout=0.0))
"""
model.add(LSTM(50,
               return_sequences=False,
               stateful=True))
"""
model.add(Dense(1))
opt=Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.)
model.compile(loss='mse', optimizer=opt)   #'rmsprop')
model.summary()
#model.load_weights('params_model_lstm_epoch_2900.hdf5')

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
    # save weights every epoch
    #model.save_weights(
    #      'params_model_lstm_epoch_{0:03d}.hdf5'.format(i), True)
    
    if i%10==0:
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
        plt.ylim(-120, 120)
        #plt.ylim(-25, 25)
        #plt.xlim(0, 260)
        plt.subplot(3, 1, 2)
        plt.plot(expected_output1)
        plt.ylim(-120, 120)
        #plt.ylim(-25, 25)
        #plt.xlim(0, 260)

        plt.title('Expected')
        plt.subplot(3, 1, 2)
        plt.plot(predicted_output)
        plt.ylim(-120, 120)
        #plt.ylim(-25, 25)
        #plt.xlim(0, 260)
        plt.title('Predicted')
        #plt.pause(3)
        #plt.savefig('plot_epoch_{0:03d}_lstm.png'.format(i), dpi=60)
        #plt.close()
        
        
        plt.subplot(3, 1, 3)
        for j in range(len(cos)):
            output1[j, 0]=expected_output1[j, 0]/expected_output1[np.argmax(expected_output1)] 
        plt.plot(output1)
        #plt.ylim(-120, 120)
        plt.subplot(3, 1, 3)
        for j in range(len(cos)):
            output[j, 0]=predicted_output[j, 0]/predicted_output[np.argmax(predicted_output)] 
        plt.plot(output)
        #plt.ylim(-120, 120)
        
        plt.pause(3)
        plt.savefig('plot_epoch_{0:03d}_lstm.png'.format(i), dpi=60)
        plt.close()
    else:
        continue
         
