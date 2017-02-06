import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from keras.utils.visualize_util import plot

import plotly.offline as py
import plotly.graph_objs as go

# Data
data = np.array([
  [0, 0, 0],
  [1, 1, 0],
  [2, 2, 0],
  [3, 3, 0],
  [4, 4, 0],
  [5, 5, 1],
  [6, 6, 1],
  [7, 7, 1],
  [8, 8, 1],
  [9, 9, 1],
])

data = np.vstack((data, data, data, data))  # Repeat data for sufficient input
data = pd.DataFrame(data, columns=['x', 'y', 'class'])

# Split X and y
X = data[['x', 'y']].values
y = data['class'].values

# Constants
batch_size = 128
hidden_layers_dim = 100
dropout_ratio = 0.2
num_epochs = 200
verbose = 0

print('batch_size: ', batch_size)
print('hidden_layers_dim: ', hidden_layers_dim)
print('dropout: ', dropout_ratio)
print('num_epochs: ', num_epochs)
print('verbose: ', verbose)
print()

# Get dimensions of input and output
input_dim = X.shape[1]
output_dim = np.max(y) + 1
print('input_dim: ', input_dim)
print('output_dim: ', output_dim)

# One-hot encoding of the class labels.
y = np_utils.to_categorical(y, output_dim)

# Create model.
# Note: Adding dropout layers helps prevent overfitting. Dropout consists in
#       randomly setting a fraction p of input units to 0 at each update 
#       during training time.
model = Sequential()
model.add(Dense(hidden_layers_dim,
                input_dim=input_dim, init='uniform', activation='relu'))
model.add(Dropout(dropout_ratio))
model.add(Dense(hidden_layers_dim, init='uniform', activation='relu'))
model.add(Dropout(dropout_ratio))
model.add(Dense(output_dim, init='uniform', activation='softmax'))

# For a multi-class classification problem
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

# Plot model
plot(model, show_shapes=True, to_file='model.png')


# Train
# The input data is shuffled at each epoch
hist = model.fit(
  X, y,
  validation_split=0.2,
  batch_size=batch_size, nb_epoch=num_epochs, verbose=verbose)

loss = hist.history['loss']
acc = hist.history['acc']
epochs = range(num_epochs)

trace0 = go.Scatter(x=epochs, y=loss, name='Loss')
trace1 = go.Scatter(x=epochs, y=acc, name='Accuracy')

layout = go.Layout(showlegend=True, title='Loss & Accuracy')
fig = go.Figure(data=[trace0, trace1], layout=layout)

py.plot(fig,
        filename='metrics.html',
        auto_open=False,
        link_text=False)


# Evaluate
loss, accuracy = model.evaluate(X, y, verbose=verbose)
print('loss: ', loss)
print('accuracy: ', accuracy)
print()

# Predict from the training set (this is bad...)
print('prediction of [1, 1]: ',
      model.predict_classes(np.array([[1, 1]]), verbose=verbose))
print('prediction of [8, 8]: ',
      model.predict_classes(np.array([[8, 8]]), verbose=verbose))

# Plot input data
class0 = data[data['class'] == 0]
class1 = data[data['class'] == 1]
trace0 = go.Scatter(x=class0['x'], y=class0['y'], name='Data (class 0)',
                    mode='lines+markers', marker={'color': 'grey'})

trace1 = go.Scatter(x=class1['x'], y=class1['y'], name='Data (class 1)',
                    mode='lines+markers', marker={'color': 'blue'})

layout = go.Layout(showlegend=True, title='Data')
fig = go.Figure(data=[trace0, trace1], layout=layout)
py.plot(fig,
        filename='data.html',
        auto_open=False,
        link_text=False)

# Plot results (correct and incorrect)
results = pd.DataFrame(data)
results['class'] = model.predict_classes(X, verbose=0)

results['class'] = ['Error' if is_error
                    else 'Correct'
                    for is_error in data['class'] != results['class']]

correct = results[results['class'] == 'Correct']
incorrect = results[results['class'] == 'Incorrect']

trace0 = go.Scatter(x=correct['x'], y=correct['y'], name='Correct predictions',
                    mode='lines+markers', marker={'color': 'green'})

trace1 = go.Scatter(x=incorrect['x'], y=incorrect['y'],
                    name='Incorrect predictions)',
                    mode='lines+markers', marker={'color': 'red'})

layout = go.Layout(showlegend=True, title='Predictions')
fig = go.Figure(data=[trace0, trace1], layout=layout)

py.plot(fig,
        filename='predictions.html',
        auto_open=False,
        link_text=False)
