"""
In this model, we stack 3 LSTM layers on top of each other, 
making the model capable of learning higher-level temporal representations.
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

num_epochs = 5

# generate dummy training data
data_dim = 16
timesteps = 8
nb_classes = 10

x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, nb_classes))

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
# returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  
# returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  
# return a single vector of dimension 32
model.add(LSTM(32))  
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



# generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, nb_classes))

model.fit(x_train, y_train,
          batch_size=64, nb_epoch=num_epochs,
          validation_data=(x_val, y_val))