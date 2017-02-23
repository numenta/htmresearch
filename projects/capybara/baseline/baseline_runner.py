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
from sklearn.metrics import accuracy_score

from baseline_utils import (create_model, predictions_vote,
                            convert_to_one_hot, save_keras_model,
                            load_keras_model, convert_to_sdrs, moving_average)

CHUNK_SIZE = 2048
BATCH_SIZE = 32
NUM_EPOCHS = 200
VERBOSE = 0
MA_WINDOW = 10
VOTE_WINDOW = 125  # Needs to be an odd number to break ties
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



def _process_chunk(chunk, output_dim):
  tmPredictedActiveCellsNZ = chunk.tmPredictedActiveCells.values
  tmPredictedActiveCells = convert_to_sdrs(tmPredictedActiveCellsNZ,
                                           NUM_TM_CELLS)
  X = moving_average(tmPredictedActiveCells, MA_WINDOW)
  y_labels = chunk.label.values
  y = convert_to_one_hot(y_labels, output_dim)
  return X, y



def _train_and_save_model(model_path, input_dim, output_dim,
                          model_history_path):
  """
  Train model, save train history and trained model.
  
  :param model_path: (str) path to serialized model.
  :param input_dim: (int) input layer dimension.
  :param output_dim: (int) output layer dimension.
  :param model_history_path: (str) path to model train history.
  :return model: (keras.Model) trained model.
  """
  start = time.time()
  model = create_model(input_dim, output_dim)

  with open(model_history_path, 'a') as historyFile:
    historyWriter = csv.writer(historyFile)
    historyWriter.writerow(['epoch', 'acc', 'loss'])

    for epoch in range(NUM_EPOCHS):
      print 'Epoch: %s/%s' % (epoch, NUM_EPOCHS)
      train_file = os.path.join(TRACE_DIR, 'trace_%s_train.csv' % EXP_NAME)
      chunks = pd.read_csv(train_file, iterator=True,
                           chunksize=CHUNK_SIZE,
                           converters={'tmPredictedActiveCells': json.loads},
                           usecols=['t', 'label', 'scalarValue',
                                    'tmPredictedActiveCells'])
      chunk_counter = 0
      for chunk in chunks:
        X, y = _process_chunk(chunk, output_dim)
        hist = model.fit(X, y, batch_size=BATCH_SIZE, shuffle=False,
                         verbose=VERBOSE, nb_epoch=1)
        acc = hist.history['acc']
        loss = hist.history['loss']
        assert len(acc) == 1  # Should be only one epoch
        historyWriter.writerow([epoch, acc[0], loss[0]])

        # Print elapsed time and row id.
        now = int(time.time() - start)
        row_id = CHUNK_SIZE * chunk_counter
        print '-> Elapsed train time: %ss - Row: %s' % (now, row_id)
        chunk_counter += 1

  save_keras_model(model, model_path)
  print 'Trained model saved:', model_path
  print 'Training history saved:', model_history_path
  return model



def _test_model(model, predictions_history_path):
  """
  Evaluate model on test set and save prediction history.
  
  :param model: (keras.Model) trained model.
  :param predictions_history_path: (str) path to prediction history file.
  :return: 
  """

  start = time.time()
  test_file = os.path.join(TRACE_DIR, 'trace_%s_test.csv' % EXP_NAME)
  chunks = pd.read_csv(test_file, iterator=True, chunksize=CHUNK_SIZE,
                       converters={'tmPredictedActiveCells': json.loads},
                       usecols=['t', 'label', 'scalarValue',
                                'tmPredictedActiveCells'])

  with open(predictions_history_path, 'a') as f:
    pred_writer = csv.writer(f)
    pred_writer.writerow(['t', 'scalar_value', 'y_pred', 'y_vote', 'y_true'])

    chunk_counter = 0
    for chunk in chunks:
      t = chunk.t.values
      X_values = chunk.scalarValue.values
      y_true = chunk.label.values
      tmPredictedActiveCellsNZ = chunk.tmPredictedActiveCells.values
      tmPredictedActiveCells = convert_to_sdrs(tmPredictedActiveCellsNZ,
                                               NUM_TM_CELLS)
      X = moving_average(tmPredictedActiveCells, MA_WINDOW)
      y_pred = model.predict_classes(X)
      y_vote = predictions_vote(y_pred, VOTE_WINDOW)

      for i in range(len(y_pred)):
        pred_writer.writerow([t[i], X_values[i], y_pred[i],
                              y_vote[i], y_true[i]])

      now = int(time.time() - start)
      row_id = CHUNK_SIZE * chunk_counter
      print 'Elapsed test time: %ss - Row: %s' % (now, row_id)
      chunk_counter += 1

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
    model = _train_and_save_model(model_path, input_dim, output_dim,
                                  model_history_path)
  else:
    model = load_keras_model(model_path)

  # Test
  _test_model(model, prediction_history_path)



if __name__ == "__main__":
  main()
