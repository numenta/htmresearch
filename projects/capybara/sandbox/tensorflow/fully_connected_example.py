# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import tensorflow as tf

# Params
input_width = 2
num_labels = 2
learning_rate = 0.01
num_iterations = 100000
print_period = 100

# Input data

# Gaussian clusters
# import random as rd
# import numpy as np
# num_examples = 40
# X = []
# y = []
# for l in range(num_labels):
#   for _ in range(num_examples / num_labels):
#     point = [l * 3 + rd.random() for k in range(input_width)] 
#     X.append(point)
#     y.append(l)
# X = np.array(X)
# y = np.array(y)
# p = np.random.permutation(len(X))
# X, y = X[p], y[p]

# XOR
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    # None that the batch size can be whatever.
    X_ = tf.placeholder(tf.float32, [None, input_width])
    y_ = tf.placeholder(tf.int32, [None])

    # Only one fully connected layer. 
    # We have 'num_labels' output labels (so 'num_labels' output nodes)
    # If you want a deep net, just repeat the fully connected layer.
    # Generates logits of size [None, num_labels] (None is the batch size)
    # For classification tasks ReLU is the preferred activation function.
    # That's the real output, and that's what we pass to the loss function 
    # to optimize. The other layers outputs, like softmax, are just for our 
    # convenience, to get human readable predictions.
    h1 = tf.contrib.layers.fully_connected(X_, num_labels, tf.nn.relu)
    h2 = tf.contrib.layers.fully_connected(h1, num_labels, tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(h2, num_labels, tf.nn.relu)
    
    # Convert logits to one-sum vector.
    # Shape [None, num_labels], type float.
    predictions = tf.nn.softmax(logits)
    
    # Convert one-hot vector to label index (int). 
    # Shape: [None]. I.e. a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(predictions, 1)
    
    # Define the loss function. 
    # Cross-entropy is a good choice for classification.
    # Goal: minimize loss function.
    # "Sparse" is useful when you know only 1 class is right.
    # You could use the non-sparse cross entropy is you want more than 1 value.
    loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_))

    # Create training op. Update weughts.
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # And, finally, an initialization op to execute before training.
    init = tf.initialize_all_variables()
    
print("images_flat: ", X_)
print("logits: ", logits)
print("loss: ", loss)
print("predictions: ", predictions)
print("predicted_labels: ", predicted_labels)


# Train
session = tf.Session(graph=graph)
session.run([init])

for i in range(num_iterations):
    _, loss_value = session.run([train, loss], 
                                feed_dict={X_: X, y_: y})
    if i % print_period == 0:
        print("Loss: ", loss_value)
        
        
# Predict
sample_X = X
predicted = session.run([predicted_labels], 
                        feed_dict={X_: sample_X})[0]
print(sample_X)
print(predicted)