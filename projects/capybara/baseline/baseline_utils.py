# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
import copy
import numpy as np
import os

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.models import load_model



def generate_sdr(n, w):
  """
  Generate a random n-dimensional SDR with w bits active
  """
  sdr = np.zeros((n,))
  random_order = np.random.permutation(np.arange(n))
  active_bits = random_order[:w]
  sdr[active_bits] = 1
  return sdr



def corrupt_sparse_vector(sdr, noise_level):
  """
  Add noise to sdr by turning off num_noise_bits active bits and turning on
  num_noise_bits in active bits
  :param sdr: (array) Numpy array of the  SDR
  :param noise_level: (float) amount of noise to be applied on the vector.
  """
  num_noise_bits = int(noise_level * np.sum(sdr))
  if num_noise_bits <= 0:
    return sdr
  active_bits = np.where(sdr > 0)[0]
  inactive_bits = np.where(sdr == 0)[0]

  turn_off_bits = np.random.permutation(active_bits)
  turn_on_bits = np.random.permutation(inactive_bits)
  turn_off_bits = turn_off_bits[:num_noise_bits]
  turn_on_bits = turn_on_bits[:num_noise_bits]

  sdr[turn_off_bits] = 0
  sdr[turn_on_bits] = 1



def generate_sdrs(num_sdr_classes, num_sdr_per_class, n, w, noise_level):
  sdrs = []
  class_ids = []
  for class_id in range(num_sdr_classes):
    class_ids.append(class_id)
    template_sdr = generate_sdr(n, w)
    sdr_cluster = []
    for _ in range(num_sdr_per_class):
      noisy_sdr = copy.copy(template_sdr)
      corrupt_sparse_vector(noisy_sdr, noise_level)
      sdrs.append(noisy_sdr)
      sdr_cluster.append(noisy_sdr)
  return sdrs, class_ids



def convert_to_one_hot(y_labels, output_dim):
  return np_utils.to_categorical(y_labels, output_dim)



def create_model(input_dim, output_dim):
  # Create model.
  model = Sequential()
  model.add(Dense(output_dim,
                  input_dim=input_dim,
                  W_regularizer=l2(.01),
                  init='uniform',
                  activation='softmax'))

  # For a multi-class classification problem
  model.compile(loss='categorical_crossentropy',
                optimizer='sgd', metrics=['accuracy'])

  return model



def train(model, X_train, y_train, X_val, y_val, batch_size, num_epochs,
          verbose):
  # The input data is shuffled at each epoch
  hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size, nb_epoch=num_epochs, verbose=verbose)
  return hist.history



def convert_to_sdr(patternNZ, input_width):
  sdr = np.zeros(input_width)
  sdr[np.array(patternNZ, dtype='int')] = 1
  return sdr



def convert_to_sdrs(patternNZs, input_width):
  sdrs = []
  for patternNZ in patternNZs:
    sdr = np.zeros(input_width, dtype='int32')
    sdr[patternNZ] = 1
    sdrs.append(sdr)
  return sdrs



def filter_sdr_columns(x):
  if type(x) == str:
    if x == '[]':
      x = []
    else:
      x = map(int, x[1:-1].split(','))
  return x



def predictions_vote(y_pred, vote_window=11):
  """
  Take the most common label over a voting window.
  
  :param y_pred: (np.array) class predictions
  :param vote_window: (int) size of the voting window
  :return: (np.array) prediction votes
  """
  n = len(y_pred)
  if vote_window > n:
    vote_window = n
    
  # Last bin vote
  last_bin_predictions = [int(y_pred[k]) for k in range(n - vote_window, n)]
  last_bin_counts = np.bincount(last_bin_predictions)
  last_bin_vote = np.argmax(last_bin_counts)

  # Rolling vote window
  votes = []
  for i in range(n):
    if i < n - vote_window:
      predictions = [int(y_pred[k]) for k in range(i, i + vote_window)]
      counts = np.bincount(predictions)
      vote = np.argmax(counts)
    else:
      vote = last_bin_vote
    votes.append(vote)
  return np.array(votes)



def save_keras_model(model, model_path):
  if os.path.exists(model_path):
    os.remove(model_path)
  model.save(model_path)
  print("Saved model to disk")



def load_keras_model(model_name):
  loaded_model = load_model("%s.h5" % model_name)
  print("Loaded model from disk")
  return loaded_model

