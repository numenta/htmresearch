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
import argparse
import csv
import json
import os
import pandas as pd
import time
import numpy as np
import yaml

from baseline_utils import (create_model,
                            convert_to_one_hot, save_keras_model,
                            load_keras_model)


def _get_union(df, ma_window, column_name):
  window = [df[column_name].values[i] for i in range(ma_window)]
  padding_value = np.mean(window, axis=0)
  union = [padding_value for _ in range(ma_window)]
  for k in range(ma_window, len(df)):
    window = [df[column_name].values[k - ma_window + i]
              for i in range(ma_window)]
    union.append(np.mean(window, axis=0))
  return np.array(union)



def _convert_df(df, y_dim, ma_window, column_name):

  if ma_window > 0:
    X = _get_union(df, ma_window, column_name)
  else:
    X = df[column_name].values

  y_labels = df.label.values
  y = convert_to_one_hot(y_labels, y_dim)
  t = df.t.values
  X_values = df.scalarValue.values
  return t, X, X_values, y, y_labels



def _convert_patternNZ_json_string_to_sdr(patternNZ_json_string, sdr_width):
  patternNZ = np.array(json.loads(patternNZ_json_string), dtype=int)
  sdr = np.zeros(sdr_width)
  sdr[patternNZ] = 1
  return sdr



def _sdr_converter(sdr_width):
  return lambda x: _convert_patternNZ_json_string_to_sdr(x, sdr_width)



def _train_on_chunks(model, output_dim, input_dim, input_file,
                     history_writer, ma_window, chunk_size,
                     batch_size, num_epochs, input_name):
  """
  Don't load all the data in memory. Read it chunk by chunk to train the model.
  :param model: (keras.Model) model to train. 
  :param output_dim: (int) dimension of the output layer.
  :param input_dim: (int) dimension of the input layer.
  :param input_file: (str) path to the input training set.
  :param history_writer: (csv.writer) file writer.
  """
  start = time.time()
  for epoch in range(num_epochs):
    print 'Epoch %s/%s' % (epoch, num_epochs)

    # Note: http://stackoverflow.com/a/1271353
    df_generator = pd.read_csv(
      input_file, chunksize=chunk_size, iterator=True,
      converters={
        'tmPredictedActiveCells': _sdr_converter(2048 * 32),
        'tmActiveCells': _sdr_converter(2048 * 32),
        'spActiveColumns': _sdr_converter(2048)
      },
      usecols=['t', 'label', 'scalarValue', 'spActiveColumns',
               'tmActiveCells', 'tmPredictedActiveCells'])

    chunk_counter = 0
    for df in df_generator:
      t, X, X_values, y, y_labels = _convert_df(df, output_dim, ma_window, 
                                                input_name)
      hist = model.fit(X, y, validation_split=0.0,
                       batch_size=batch_size, shuffle=False,
                       verbose=0, nb_epoch=1)
      acc = hist.history['acc']
      loss = hist.history['loss']
      assert len(acc) == 1  # Should be only one epoch
      history_writer.writerow([epoch, acc[0], loss[0]])
      chunk_counter += 1

      # Print elapsed time and # of rows processed.
      now = int(time.time() - start)
      row_id = chunk_size * chunk_counter
      print '-> Elapsed train time: %ss - Rows processed: %s' % (now, row_id)



def _train(model, output_dim, input_dim, input_file, history_writer,
           ma_window, chunk_size, batch_size, num_epochs, input_name):
  """
  Load all the data in memory and train the model.
  :param model: (keras.Model) model to train. 
  :param output_dim: (int) dimension of the output layer.
  :param input_dim: (int) dimension of the input layer.
  :param input_file: (str) path to the input training set.
  :param history_writer: (csv.writer) file writer.
  """
  start = time.time()
  df = pd.read_csv(
    input_file,
    converters={
      'tmPredictedActiveCells': _sdr_converter(2048 * 32),
      'tmActiveCells': _sdr_converter(2048 * 32),
      'spActiveColumns': _sdr_converter(2048)
    },
    usecols=['t', 'label', 'scalarValue', 'spActiveColumns',
             'tmActiveCells', 'tmPredictedActiveCells'])
  t, X, X_values, y, y_labels = _convert_df(df, output_dim, ma_window, input_name)

  hist = model.fit(X, y, validation_split=0.0,
                   batch_size=batch_size, shuffle=False,
                   verbose=1, nb_epoch=num_epochs)
  acc = hist.history['acc']
  loss = hist.history['loss']
  for epoch in range(num_epochs):
    history_writer.writerow([epoch, acc[epoch], loss[epoch]])
  print 'Elapsed time: %s' % (time.time() - start)



def _train_and_save_model(model_path, model_history_path,
                          input_dim, output_dim, lazy, train_file,
                          ma_window, chunk_size, batch_size, num_epochs, 
                          input_name):
  """
  Train model, save train history and trained model.
  
  :param model_path: (str) path to serialized model.
  :param model_history_path: (str) path to model train history.
  :param input_dim: (int) input layer dimension.
  :param output_dim: (int) output layer dimension.
  :param lazy: (bool) whether to load the whole input file in memory or to 
    read it lazily in chunks. 
  :return model: (keras.Model) trained model.
  """
  model = create_model(input_dim, output_dim)

  with open(model_history_path, 'a') as historyFile:
    history_writer = csv.writer(historyFile)
    history_writer.writerow(['epoch', 'acc', 'loss'])

    if lazy:
      _train_on_chunks(model, output_dim, input_dim, train_file,
                       history_writer, ma_window, chunk_size,
                       batch_size, num_epochs, input_name)
    else:
      _train(model, output_dim, input_dim, train_file, history_writer,
             ma_window, chunk_size, batch_size, num_epochs, input_name)

  save_keras_model(model, model_path)
  print 'Trained model saved:', model_path
  print 'Training history saved:', model_history_path
  return model



def _test_model(model, predictions_history_path, input_dim, output_dim,
                test_file, chunk_size, ma_window, input_name):
  """
  Evaluate model on test set and save prediction history.
  
  :param model: (keras.Model) trained model.
  :param predictions_history_path: (str) path to prediction history file.
  :param input_dim: (int) input layer dimension.
  :param output_dim: (int) output layer dimension.
  """

  start = time.time()
  chunks = pd.read_csv(
    test_file, iterator=True, chunksize=chunk_size,
    converters={
      'tmPredictedActiveCells': _sdr_converter(2048 * 32),
      'tmActiveCells': _sdr_converter(2048 * 32),
      'spActiveColumns': _sdr_converter(2048)
    },
    usecols=['t', 'label', 'scalarValue', 'spActiveColumns',
             'tmActiveCells', 'tmPredictedActiveCells'])

  with open(predictions_history_path, 'a') as f:
    pred_writer = csv.writer(f)
    pred_writer.writerow(['t', 'scalar_value', 'y_pred', 'y_true'])

    chunk_counter = 0
    for chunk in chunks:
      t, X, X_values, y, y_labels = _convert_df(chunk, output_dim, ma_window, input_name)
      y_pred = model.predict_classes(X)
      y_true = y_labels

      for i in range(len(y_pred)):
        pred_writer.writerow([t[i], X_values[i], y_pred[i], y_true[i]])

      now = int(time.time() - start)
      row_id = chunk_size * chunk_counter
      print '\nElapsed test time: %ss - Row: %s' % (now, row_id)
      chunk_counter += 1

  print 'Elapsed time: %ss' % (time.time() - start)
  print 'Test prediction history saved:', predictions_history_path



def _getConfig(configFilePath):
  with open(configFilePath, 'r') as ymlFile:
    config = yaml.load(ymlFile)

  input_dir = config['inputs']['input_dir']
  train_file_name = config['inputs']['train_file_name']
  test_file_name = config['inputs']['test_file_name']
  metric_name = config['inputs']['metric_name']
  results_output_dir = config['outputs']['results_output_dir']
  model_output_dir = config['outputs']['model_output_dir']
  history_file = config['outputs']['history_file']
  prediction_file = config['outputs']['prediction_file']
  model_name = config['outputs']['model_name']
  chunk_size = config['params']['chunk_size']
  batch_size = config['params']['batch_size']
  num_epochs = config['params']['num_epochs']
  ma_window = config['params']['ma_window']
  input_dim = config['params']['input_dim']
  output_dim = config['params']['output_dim']
  labels = config['params']['labels']
  lazy = config['params']['lazy']
  train = config['params']['train']

  return (input_dir,
          train_file_name,
          test_file_name,
          metric_name,
          results_output_dir,
          model_output_dir,
          history_file,
          prediction_file,
          model_name,
          chunk_size,
          batch_size,
          num_epochs,
          ma_window,
          input_dim,
          output_dim,
          labels,
          lazy,
          train)



def main():
  # Get input args
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', '-c',
                      dest='config',
                      type=str,
                      default='configs/uci.yml',
                      help='Name of YAML config file.')
  options = parser.parse_args()
  configFile = options.config

  # Get config options.
  (input_dir,
   train_file_name,
   test_file_name,
   metric_name,
   results_output_dir,
   model_output_dir,
   history_file,
   prediction_file,
   model_name,
   chunk_size,
   batch_size,
   num_epochs,
   ma_window,
   input_dim,
   output_dim,
   labels,
   lazy,
   train) = _getConfig(configFile)

  train_file = os.path.join(input_dir, train_file_name)
  test_file = os.path.join(input_dir, test_file_name)

  # Model dimensions
  print 'input_dim', input_dim
  print 'output_dim', output_dim
  print 'train', train
  print ''

  # Make sure output directories exist
  if not os.path.exists(results_output_dir):
    os.makedirs(results_output_dir)
  if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)
  model_path = os.path.join(model_output_dir, model_name)

  # Clean model history 
  model_history_path = os.path.join(results_output_dir, history_file)
  if os.path.exists(model_history_path):
    os.remove(model_history_path)

  # Clean predictions history    
  prediction_history_path = os.path.join(results_output_dir, prediction_file)
  if os.path.exists(prediction_history_path):
    os.remove(prediction_history_path)

  # Train
  if train:
    model = _train_and_save_model(model_path, model_history_path,
                                  input_dim, output_dim, lazy, train_file,
                                  ma_window, chunk_size, batch_size,
                                  num_epochs, metric_name)
  else:
    model = load_keras_model(model_path)

  # Test
  _test_model(model, prediction_history_path, input_dim, output_dim, test_file,
              chunk_size, ma_window, metric_name)



if __name__ == "__main__":
  main()
