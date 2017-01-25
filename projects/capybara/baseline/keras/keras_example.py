# Data pre-processing imports 
import pandas as pd
import numpy as np
from keras.utils import np_utils
import seaborn as sns

# Keras imports
from keras.models import Sequential
from keras.layers.core import Dense, Dropout

# Constants
batch_size = 128
hidden_layers_dim = 100
dropout_ratio = 0.2
num_epochs = 100
verbose = 0

print('batch_size: ', batch_size)
print('hidden_layers_dim: ', hidden_layers_dim)
print('dropout: ', dropout_ratio)
print('num_epochs: ', num_epochs)
print('verbose: ', verbose)
print()

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
                input_dim=input_dim, init='uniform', activation='tanh'))
model.add(Dropout(dropout_ratio))
model.add(Dense(hidden_layers_dim, init='uniform', activation='tanh'))
model.add(Dropout(dropout_ratio))
model.add(Dense(output_dim, init='uniform', activation='softmax'))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# Train
model.fit(
  X, y,
  validation_split=0.2,
  batch_size=batch_size, nb_epoch=num_epochs, verbose=verbose)

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

# Plot
sns.lmplot('x', 'y', data, 'class', fit_reg=False).set(title='Data')
results = pd.DataFrame(data.copy())
results['class'] = model.predict_classes(X, verbose=0)
sns.lmplot('x', 'y', results, 'class',
           fit_reg=False).set(title='Trained Result')
results['class'] = ['Error' if is_error
                    else 'Non Error'
                    for is_error in data['class'] != results['class']]
sns.lmplot('x', 'y', results, 'class', fit_reg=False).set(title='Errors')

sns.plt.show()
