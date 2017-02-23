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
if __name__ == "__main__":
  start = time.time()

  # Make sure output directories exist
  if not os.path.exists(RESULTS_OUTDIR):
    os.makedirs(RESULTS_OUTDIR)
  if not os.path.exists(MODEL_OUTDIR):
    os.makedirs(MODEL_OUTDIR)
  model_path = os.path.join(MODEL_OUTDIR, MODEL_NAME)

  # Get input args
  parser = argparse.ArgumentParser()
  parser.add_argument('--skip-train', '-st',
                      dest='skip_train',
                      action='store_true',
                      default=False)
  options = parser.parse_args()
  TRAIN = not options.skip_train
  print 'TRAIN', TRAIN

  # Train model or load from disk 
  if TRAIN:
    # Create model 
    input_dim = NUM_TM_CELLS
    output_dim = len(LABELS)
    print 'INPUT_DIM', input_dim
    print 'OUTPUT_DIM', output_dim
    model = create_model(input_dim, output_dim)

    # Save model train history
    history_path = os.path.join(RESULTS_OUTDIR, HISTORY_FILE)
    if os.path.exists(history_path):
      os.remove(history_path)

    with open(history_path, 'a') as historyFile:
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
          now = int(time.time() - start)
          row_id = CHUNK_SIZE * chunk_counter
          print '-> Elapsed time: %ss - Row: %s' % (now, row_id)
          tmPredictedActiveCellsNZ = chunk.tmPredictedActiveCells.values
          tmPredictedActiveCells = convert_to_sdrs(tmPredictedActiveCellsNZ,
                                                   NUM_TM_CELLS)
          X = moving_average(tmPredictedActiveCells, MA_WINDOW)

          # One-hot encoding of the class labels.
          y_labels = chunk.label.values
          y = convert_to_one_hot(y_labels, output_dim)
          hist = model.fit(X, y, batch_size=BATCH_SIZE, shuffle=False,
                           verbose=VERBOSE, nb_epoch=1)

          acc = hist.history['acc']
          loss = hist.history['loss']
          assert len(acc) == 1  # Should be only one epoch
          historyWriter.writerow([epoch, acc[0], loss[0]])
          chunk_counter += 1 

    save_keras_model(model, model_path)
  else:
    model = load_keras_model(model_path)

  # Evaluate model on test set
  test_file = os.path.join(TRACE_DIR, 'trace_%s_test.csv' % EXP_NAME)
  chunks = pd.read_csv(test_file, iterator=True, chunksize=CHUNK_SIZE,
                       converters={'tmPredictedActiveCells': json.loads},
                       usecols=['t', 'label', 'scalarValue',
                                'tmPredictedActiveCells'])

  predictions_path = os.path.join(RESULTS_OUTDIR, PREDICTIONS_FILE)
  if os.path.exists(predictions_path):
    os.remove(predictions_path)

  with open(predictions_path, 'a') as f:
    pred_writer = csv.writer(f)
    pred_writer.writerow(['t', 'scalar_value', 'y_pred', 'y_vote', 'y_true'])

    chunks_counter = 0
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

      acc = accuracy_score(y_true, y_pred)
      now = int(time.time() - start)
      row_id = CHUNK_SIZE * chunk_counter
      print 'Elapsed time: %ss - Row: %s' % (now, row_id)
      chunks_counter +=1
