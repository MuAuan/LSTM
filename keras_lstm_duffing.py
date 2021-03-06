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
from scipy.integrate import odeint, simps


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 25
epochs = 3000
# number of elements ahead that are used to make the prediction
lahead = 2000

cos = np.zeros((5000, 1, 1))
def duffing(var, t, gamma, a, b, F0, omega, delta):
    """
    var = [x, p]
    dx/dt = p
    dp/dt = -gamma*p + 2*a*x - 4*b*x**3 + F0*cos(omega*t + delta)
    """
    x_dot = var[1]
    p_dot = -gamma * var[1] + 2 * a * var[0] - 4 * b * var[0]**3 + F0 * np.cos(omega * t + delta)

    return np.array([x_dot, p_dot])

# parameter
F0, gamma, omega, delta = 10, 0.1, np.pi/3, 1.5*np.pi
a, b = 1/4, 1/2
y0 = [0.5, 0]
t = np.linspace(0, 200, 5001)
sol = odeint(duffing, y0, t, args=(gamma, a, b, F0, omega, delta))

x, p = sol.T[0], sol.T[1]
plt.plot(x, p, ".", markersize=4)
plt.pause(3)
plt.savefig('plot_x-p_duffing.png', dpi=60)
plt.close()

#cos = np.zeros((5000, 1, 1))
for i in range(len(cos)):
    #cos[i,0,0]=100*x[i]
    cos[i,0,0]=100*(x[i])/x[np.argmax(x)]
  
print('Generating Data...')
cos = cos   #gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = cos[i]  #np.mean(cos[i + 1:i + 1 + 1])  #lahead =1
expected_output1 = np.zeros((len(cos), 1))
for i in range(len(cos)):
    expected_output1[i, 0] = cos[i]  #np.mean(cos[i + 1:i + 1 + 1])  #lahead =1    

print('Output shape:', expected_output.shape)

print('Creating Model...')
model = Sequential()
model.add(LSTM(200,
               input_shape=(tsteps, 1),
               batch_size=batch_size,
               return_sequences=False,
               stateful=True,
              dropout=0.5))
"""
model.add(LSTM(100,
               return_sequences=False,
               stateful=True,
              dropout=0.5))
"""
model.add(Dense(1))
opt=Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.)
model.compile(loss='mse', optimizer=opt)   #'rmsprop')
model.summary()
#model.load_weights('params_model_lstm_epoch_200.hdf5')
lr=0.0001

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
    
    if i%100==0:
        # save weights every 100 epoch
        model.save_weights(
              'params_model_duffing_epoch_{0:03d}.hdf5'.format(i), True)
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
        #plt.xlim(0, 260)
        plt.subplot(3, 1, 2)
        plt.plot(expected_output1)
        plt.ylim(-120, 120)
        #plt.xlim(0, 260)

        plt.title('Expected')
        plt.subplot(3, 1, 3)
        plt.plot(predicted_output)
        plt.ylim(-120, 120)
        #plt.xlim(0, 260)
        plt.title('Predicted')
        plt.pause(3)
        plt.savefig('plot_duffing_epoch_{0:03d}_lstm.png'.format(i), dpi=60)
        plt.close()
    else:
        continue
         
