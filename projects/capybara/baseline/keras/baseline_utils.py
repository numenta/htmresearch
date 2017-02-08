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
import csv
import copy
import itertools
import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.visualize_util import plot
from keras.regularizers import l2
from keras.models import load_model

import plotly.offline as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt

CONFIG = 'sp=True_tm=True_tp=False_SDRClassifier'
INPUT_WIDTH = 2048 * 32
ACTIVE_CELLS_WEIGHT = 0.0
PRED_ACTIVE_CELLS_WEIGHT = 1.0
MA_WINDOW = 10



def get_file_name(exp_name, network_config):
  trace_csv = os.path.join('traces', '%s_%s.csv' % (exp_name,
                                                    network_config))
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      os.pardir, os.pardir, 'classification', 'results',
                      trace_csv)



def convert_to_sdr(patternNZ, input_width):
  sdr = np.zeros(input_width)
  sdr[np.array(patternNZ, dtype='int')] = 1
  return sdr



def convert_to_sdrs(patterNZs, input_width):
  sdrs = []
  for i in range(len(patterNZs)):
    patternNZ = patterNZs[i]
    sdr = np.zeros(input_width, dtype='int32')
    sdr[patternNZ] = 1
    sdrs.append(sdr)
  return sdrs



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



def create_model(input_dim, output_dim, num_epochs, verbose):
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

  # Plot model
  plot(model, show_shapes=True, to_file='baseline_model.png')

  return model



def train(model, X_train, y_train, X_val, y_val, batch_size, num_epochs,
          verbose):
  # The input data is shuffled at each epoch
  hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size, nb_epoch=num_epochs, verbose=verbose)
  return hist.history



def plot_train_history(num_epochs, history, title):
  loss = history['loss']
  acc = history['acc']
  epochs = range(num_epochs)

  trace0 = go.Scatter(x=epochs, y=loss, name='Loss')
  trace1 = go.Scatter(x=epochs, y=acc, name='Accuracy')

  layout = go.Layout(showlegend=True, title='Loss & Accuracy (%s)' % title)
  fig = go.Figure(data=[trace0, trace1], layout=layout)

  py.plot(fig,
          filename='%s_metrics.html' % title,
          auto_open=False,
          link_text=False)



def filter_sdr_columns(x):
  if type(x) == str:
    if x == '[]':
      x = []
    else:
      x = map(int, x[1:-1].split(','))
  return x



def load_traces(file_name, start=None, end=None):
  """
  Load network traces from CSV
  :param file_name: (str) name of the file
  :return traces: (dict) network traces. E.g: activeCells, sensorValues, etc.
  """

  df = pd.read_csv(file_name)
  df = df[start:end]

  traces = dict()
  for column in df.columns.values:
    if column in ['tmPredictedActiveCells',
                  'tpActiveCells',
                  'tmActiveCells']:
      traces[column] = df[column].apply(filter_sdr_columns).values
    else:
      traces[column] = df[column].apply(lambda x: float(x)).values
  return traces



def load_sdrs(exp_name, start=None, end=None):
  # load traces
  file_name = get_file_name(exp_name, CONFIG)
  traces = load_traces(file_name, start, end)

  sensor_values = traces['sensorValue']
  categories = traces['actualCategory']
  active_cells = traces['tmActiveCells']
  predicted_active_cells = traces['tmPredictedActiveCells']

  # generate sdrs to cluster
  active_cells_sdrs = convert_to_sdrs(active_cells, INPUT_WIDTH)
  predicted_active_cells_sdrs = np.array(
    convert_to_sdrs(predicted_active_cells, INPUT_WIDTH))
  sdrs = (float(ACTIVE_CELLS_WEIGHT) * np.array(active_cells_sdrs) +
          float(PRED_ACTIVE_CELLS_WEIGHT) * predicted_active_cells_sdrs)

  union = moving_average(sdrs, MA_WINDOW)

  print 'Data loaded!'
  return sensor_values, sdrs, categories, union



def save_traces(file_path, traces, start, end):
  headers = traces.keys()
  file_name = get_file_name(file_path, CONFIG)
  with open(file_name, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for i in range(start, end):
      row = [traces[h][i] for h in headers]
      writer.writerow(row)

  print 'File saved: %s' % file_name



def plot_data(X, y_labels, t, title):
  unique_labels = np.unique(y_labels)
  print('unique labels (%s): %s' % (title, unique_labels))

  colors = ['grey', 'blue', 'black', 'orange', 'yellow', 'pink']
  # Plot input data
  traces = []
  for label in unique_labels:
    trace = go.Scatter(x=t[np.where(y_labels == label)[0]],
                       y=X[np.where(y_labels == label)[0]][:, 0],
                       name='Data (class %s)' % label,
                       mode='markers',
                       marker={'color': colors[int(label)]})

    traces.append(trace)

  layout = go.Layout(showlegend=True, title='Data (%s)' % title)
  fig = go.Figure(data=traces, layout=layout)
  py.plot(fig,
          filename='%s_data.html' % title,
          auto_open=False,
          link_text=False)



def evaluate(y_true, y_pred):
  accuracy = accuracy_score(y_true, y_pred)
  return accuracy
  
  
def predict(model, X, vote_window):
  y_pred = model.predict_classes(X)
  if vote_window:
    return predictions_vote(y_pred, vote_window)
  else:
    return y_pred


def predictions_vote(y_pred, vote_window=10):
  """
  Take the most common label over a voting window.
  
  :param y_pred: (np.array) class predictions
  :param vote_window: (int) size of the voting window
  :return: (np.array) prediction votes
  """
  if len(y_pred) < vote_window:
    raise ValueError('The number of raw predictions (%s) must be at least the '
                     'size of the vote window (%s)' % (len(y_pred),
                                                       vote_window))
  votes = []
  for i in range(len(y_pred)):
    if i < vote_window:
      vote = y_pred[i]
    else:
      predictions = [int(y_pred[k]) for k in range(i - vote_window, i)]
      counts = np.bincount(predictions)
      vote = np.argmax(counts)
    votes.append(vote)
  return np.array(votes)



def plot_predictions(X_values, t, y_true, y_pred, title):
  """
  Plot results (correct and incorrect)  
  
  :param X_values: (np.array) input scalar values (before any encoding)
  :param t: (np.array) timesteps
  """
  
  prediction_results = pd.DataFrame(y_true) == pd.DataFrame(y_pred)
  
  correct = []
  incorrect = []
  for r in prediction_results.values:
    correct.append(r[0])
    incorrect.append(not r[0])

  t_correct = t[correct]
  t_incorrect = t[incorrect]

  X_values_test_correct = X_values[correct]
  X_values_test_incorrect = X_values[incorrect]

  trace0 = go.Scatter(x=t_correct, y=X_values_test_correct[:, 0],
                      name='Correct predictions',
                      mode='markers', marker={'color': 'green'})

  trace1 = go.Scatter(x=t_incorrect, y=X_values_test_incorrect[:, 0],
                      name='Incorrect predictions',
                      mode='markers', marker={'color': 'red'})

  layout = go.Layout(showlegend=True, title='Predictions (%s)' % title)
  fig = go.Figure(data=[trace0, trace1], layout=layout)

  py.plot(fig,
          filename='%s_predictions.html' % title,
          auto_open=False,
          link_text=False)



def plot_confusion_matrix(cm,
                          filename,
                          classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  plt.figure()
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=30)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(filename)



def save_model(model, model_name):
  model.save('%s.h5' % model_name)
  print("Saved model to disk")



def load_model(model_name):
  loaded_model = load_model("%s.h5" % model_name)
  print("Loaded model from disk")
  return loaded_model



def find_labels_used(y_labels):
  df = pd.DataFrame(y_labels)
  return df[0].unique()



def online_moving_average(last_ma, new_value, moving_average_window):
  """
  Online computation of moving average.
  From: http://www.daycounter.com/LabBook/Moving-Average.phtml
  """

  ma = last_ma + (new_value - last_ma) / float(moving_average_window)
  return ma



def moving_average(a, n=10):
  ret = np.cumsum(a, axis=0, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return np.append(a[:n - 1], ret[n - 1:] / n, 0)
