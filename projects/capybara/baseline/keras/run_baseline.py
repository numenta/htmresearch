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
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from classif_report import plot_classification_report
from baseline_utils import (load_sdrs, create_model, train, plot_train_history,
                            plot_data, plot_predictions,
                            evaluate, predict, convert_to_one_hot,
                            plot_confusion_matrix,
                            save_model, load_model, find_labels_used)

LABELS = {
  1: 'WALKING',
  2: 'WALKING_UPSTAIRS',
  3: 'WALKING_DOWNSTAIRS',
  4: ' SITTING',
  5: 'STANDING',
  6: 'LAYING'
}

BATCH_SIZE = 32  # 128
NUM_EPOCHS = 200
VERBOSE = 1
VOTE_WINDOW = 125  # Needs to be an un-even number to break ties
EXP_NAME = 'body_acc_x'

if __name__ == "__main__":
  # Classification options.
  #  - sdr: classify SDRs (TM states)
  #  - union: union of SDRs (union of TM states)
  #  - raw: classify scalar data
  #  - dummy: classify a trivial signal (used for debugging)
  valid_inputs = ['union', 'sdr', 'raw', 'dummy']

  parser = argparse.ArgumentParser()
  parser.add_argument('--input', type=str, dest='input', required=True)
  parser.add_argument('--train', dest='train', action='store_true',
                      default=False)
  parser.add_argument('--plot', dest='plot', action='store_true',
                      default=False)
  options = parser.parse_args()
  INPUT = options.input
  TRAIN = options.train
  PLOT = options.plot

  print 'INPUT: %s' % INPUT
  print 'TRAIN: %s' % TRAIN
  print 'PLOT: %s' % PLOT

  signals = ['t', 'X', 'X_values', 'y', 'y_labels']
  data = {'train': {}, 'val': {}, 'test': {}}
  input_dim = None
  output_dim = None
  for phase, dataset in data.items():
    (sensor_values, sdrs, categories,
     union) = load_sdrs('%s_%s' % (phase, EXP_NAME))
    t = np.arange(0, len(sensor_values))
    X_values = np.array([[s] for s in sensor_values])
    y_labels = np.array([int(c) for c in categories])

    # Set the X signal that will be used to train the network
    if INPUT == 'sdr':
      X = np.array(sdrs)
    elif INPUT == 'union':
      X = np.array(union)
    elif INPUT == 'raw':
      X = np.array([[s] for s in sensor_values])
    elif INPUT == 'dummy':
      X = np.array([[c] for c in categories])
      X_values = X
    else:
      raise ValueError('You can only classify %s' % valid_inputs)

    # Get dimensions of input and output
    in_dim = X.shape[1]
    if not input_dim:
      input_dim = in_dim
    else:
      if input_dim != in_dim:
        raise ValueError('Input dim different from previous input dim')
    out_dim = np.max(y_labels) + 1
    if not output_dim:
      output_dim = out_dim
    else:
      if output_dim != out_dim:
        raise ValueError('Output dim different from previous output dim')

    # One-hot encoding of the class labels.
    y = convert_to_one_hot(y_labels, output_dim)

    data[phase]['t'] = t
    data[phase]['X'] = X
    data[phase]['X_values'] = X_values
    data[phase]['y'] = y
    data[phase]['y_labels'] = y_labels

  ### Data ready ###

  # Make sure length of each signal is the same in each phase
  for d in data.values():
    for i in range(len(signals)):
      for j in range(i, len(signals)):
        assert len(d[signals[i]]) == len(d[signals[j]])

  # Print some info about the network
  print 'num_epochs: %s' % NUM_EPOCHS
  print 'verbose: %s' % VERBOSE
  print 'batch_size: %s' % BATCH_SIZE
  print 'input_dim: %s' % input_dim
  print 'output_dim: %s' % output_dim

  # Create and train network
  model_name = 'model'
  if TRAIN:
    model = create_model(input_dim, output_dim, NUM_EPOCHS, VERBOSE)

    history = train(model, data['train']['X'], data['train']['y'],
                    data['val']['X'], data['val']['y'],
                    BATCH_SIZE, NUM_EPOCHS, VERBOSE)

    plot_train_history(NUM_EPOCHS, history, INPUT)

    save_model(model, model_name)
  else:
    model = load_model(model_name)

  # Plot data and results
  if PLOT:
    phases_to_plot = ['train']
    for phase, d in data.items():
      plot_data(d['X_values'], d['y_labels'], d['t'],
                '%s_%s' % (INPUT, phase))

      # Evaluate trained model: 
      print '== Evaluate %s set ==' % phase
      loss, accuracy = model.evaluate(d['X'], d['y'], verbose=0)
      print '\n--> Model loss and accuracy: %.4f' % accuracy

      # 1) With no window
      y_pred = predict(model, d['X'], None)
      accuracy_no_voting = evaluate(d['y_labels'], y_pred)
      print ('--> Accuracy (no voting window): %.4f' % accuracy_no_voting)

      # 3) With voting window = 1
      y_pred = predict(model, d['X'], 1)
      accuracy_1_vote = evaluate(d['y_labels'], y_pred)
      print ('--> Accuracy (voting window = 1): %.4f' % accuracy_1_vote)
      # assert abs(accuracy - accuracy_1_vote) <= 0.01


      # 4) With another voting window
      y_pred = predict(model, d['X'], VOTE_WINDOW)
      accuracy_w_voting = evaluate(d['y_labels'], y_pred)
      print ('--> Accuracy (voting window = %s): %.4f' % (VOTE_WINDOW,
                                                          accuracy_w_voting))

      # Plot predictions
      plot_predictions(d['X_values'], d['t'], d['y_labels'], y_pred,
                       '%s_%s' % (INPUT, phase))

      # Compute confusion matrix
      np.set_printoptions(precision=2)
      cnf_matrix = confusion_matrix(d['y_labels'], y_pred)

      # Find labels in use
      label_list = find_labels_used(d['y_labels'])
      label_list = [LABELS[l] for l in label_list]

      # Plot confusion matrices
      _ = plot_confusion_matrix(cnf_matrix,
                                '%s_%s_cnf_matrix.png' % (INPUT, phase),
                                classes=label_list,
                                normalize=True,
                                title='%s_confusion_matrix' % phase)

      clf_report = classification_report(d['y_labels'], y_pred,
                                         target_names=label_list)

      plot_classification_report(clf_report, '%s_%s_classification_report.png'
                                 % (INPUT, phase))
