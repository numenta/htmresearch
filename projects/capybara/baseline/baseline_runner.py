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

from baseline_utils import (create_model,
                            convert_to_one_hot, save_keras_model,
                            load_keras_model)

CHUNK_SIZE = 2048
BATCH_SIZE = 32
NUM_EPOCHS = 10
MA_WINDOW = 255
NUM_TM_CELLS = 2048 * 32

TRACE_DIR = '../htm/traces'
EXP_NAME = 'body_acc_x_inertial_signals'

RESULTS_OUTDIR = 'results'
MODEL_OUTDIR = 'model'
HISTORY_FILE = 'train_history.csv'
PREDICTIONS_FILE = 'predictions.csv'
MODEL_NAME = 'baseline.h5'

LABELS = ['WALKING',
          'WALKING_UPSTAIRS',
          'WALKING_DOWNSTAIRS',
          'SITTING',
          'STANDING',
          'LAYING']

# If LAZY=True, don't load all the data in memory and train the model chunk by 
# chunk. Repeat for each epoch. (Memory efficient but slower because 
# at each epoch, the lazy panda data frame iterator needs to re-created).

# If LAZY=False, load all the data in memory and train the model for 
# multiple epochs (Memory intensive, but faster - since the panda data 
# frame is only loaded at the beginning).

LAZY = False



def _convert_df(df, y_dim):
  union = df.tmPredictedActiveCells.rolling(MA_WINDOW).mean()
  union[:MA_WINDOW - 1] = df.tmPredictedActiveCells[:MA_WINDOW - 1]  # no NaNs
  X = np.array([u for u in union.values])
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



def _train_on_chunks(model, output_dim, input_dim, input_file, history_writer):
  """
  Don't load all the data in memory. Read it chunk by chunk to train the model.
  :param model: (keras.Model) model to train. 
  :param output_dim: (int) dimension of the output layer.
  :param input_dim: (int) dimension of the input layer.
  :param input_file: (str) path to the input training set.
  :param history_writer: (csv.writer) file writer.
  """
  start = time.time()
  for epoch in range(NUM_EPOCHS):
    print 'Epoch %s/%s' % (epoch, NUM_EPOCHS)

    # Note: http://stackoverflow.com/a/1271353
    df_generator = pd.read_csv(
      input_file, chunksize=CHUNK_SIZE, iterator=True,
      converters={'tmPredictedActiveCells': _sdr_converter(input_dim)},
      usecols=['t', 'label', 'scalarValue', 'tmPredictedActiveCells'])

    chunk_counter = 0
    for df in df_generator:
      t, X, X_values, y, y_labels = _convert_df(df, output_dim)
      hist = model.fit(X, y, validation_split=0.0,
                       batch_size=BATCH_SIZE, shuffle=False,
                       verbose=0, nb_epoch=1)
      acc = hist.history['acc']
      loss = hist.history['loss']
      assert len(acc) == 1  # Should be only one epoch
      history_writer.writerow([epoch, acc[0], loss[0]])
      chunk_counter += 1

      # Print elapsed time and # of rows processed.
      now = int(time.time() - start)
      row_id = CHUNK_SIZE * chunk_counter
      print '-> Elapsed train time: %ss - Rows processed: %s' % (now, row_id)



def _train(model, output_dim, input_dim, input_file, history_writer):
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
    converters={'tmPredictedActiveCells': _sdr_converter(input_dim)},
    usecols=['t', 'label', 'scalarValue', 'tmPredictedActiveCells'])
  t, X, X_values, y, y_labels = _convert_df(df, output_dim)

  hist = model.fit(X, y, validation_split=0.0,
                   batch_size=BATCH_SIZE, shuffle=False,
                   verbose=1, nb_epoch=NUM_EPOCHS)
  acc = hist.history['acc']
  loss = hist.history['loss']
  for epoch in range(NUM_EPOCHS):
    history_writer.writerow([epoch, acc[epoch], loss[epoch]])
  print 'Elapsed time: %s' % (time.time() - start)



def _train_and_save_model(model_path, model_history_path,
                          input_dim, output_dim, lazy=False):
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
    train_file = os.path.join(TRACE_DIR, 'trace_%s_train.csv' % EXP_NAME)

    if lazy:
      _train_on_chunks(model, output_dim, input_dim, train_file,
                       history_writer)
    else:
      _train(model, output_dim, input_dim, train_file, history_writer)

  save_keras_model(model, model_path)
  print 'Trained model saved:', model_path
  print 'Training history saved:', model_history_path
  return model



def _test_model(model, predictions_history_path, input_dim, output_dim):
  """
  Evaluate model on test set and save prediction history.
  
  :param model: (keras.Model) trained model.
  :param predictions_history_path: (str) path to prediction history file.
  :param input_dim: (int) input layer dimension.
  :param output_dim: (int) output layer dimension.
  """

  start = time.time()
  test_file = os.path.join(TRACE_DIR, 'trace_%s_test.csv' % EXP_NAME)
  chunks = pd.read_csv(
    test_file, iterator=True, chunksize=CHUNK_SIZE,
    converters={'tmPredictedActiveCells': _sdr_converter(input_dim)},
    usecols=['t', 'label', 'scalarValue', 'tmPredictedActiveCells'])

  with open(predictions_history_path, 'a') as f:
    pred_writer = csv.writer(f)
    pred_writer.writerow(['t', 'scalar_value', 'y_pred', 'y_true'])

    chunk_counter = 0
    for chunk in chunks:
      t, X, X_values, y, y_labels = _convert_df(chunk, output_dim)
      y_pred = model.predict_classes(X)
      y_true = y_labels

      for i in range(len(y_pred)):
        pred_writer.writerow([t[i], X_values[i], y_pred[i], y_true[i]])

      now = int(time.time() - start)
      row_id = CHUNK_SIZE * chunk_counter
      print 'Elapsed test time: %ss - Row: %s' % (now, row_id)
      chunk_counter += 1

  print 'Elapsed time: %ss' % (time.time() - start)
  print 'Test prediction history saved:', predictions_history_path



def main():
  # Make sure output directories exist
  if not os.path.exists(RESULTS_OUTDIR):
    os.makedirs(RESULTS_OUTDIR)
  if not os.path.exists(MODEL_OUTDIR):
    os.makedirs(MODEL_OUTDIR)
  model_path = os.path.join(MODEL_OUTDIR, MODEL_NAME)

  # Clean model history 
  model_history_path = os.path.join(RESULTS_OUTDIR, HISTORY_FILE)
  if os.path.exists(model_history_path):
    os.remove(model_history_path)

  # Clean predictions history    
  prediction_history_path = os.path.join(RESULTS_OUTDIR, PREDICTIONS_FILE)
  if os.path.exists(prediction_history_path):
    os.remove(prediction_history_path)

  # Get input args
  parser = argparse.ArgumentParser()
  parser.add_argument('--skip-train', '-st',
                      dest='skip_train',
                      action='store_true',
                      default=False)
  options = parser.parse_args()
  train = not options.skip_train

  # Model dimensions
  input_dim = NUM_TM_CELLS
  output_dim = len(LABELS)
  print 'INPUT_DIM', input_dim
  print 'OUTPUT_DIM', output_dim
  print 'TRAIN', train

  # Train
  if train:
    model = _train_and_save_model(model_path, model_history_path,
                                  input_dim, output_dim, lazy=LAZY)
  else:
    model = load_keras_model(model_path)

  # Test
  _test_model(model, prediction_history_path, input_dim, output_dim)



if __name__ == "__main__":
  main()
