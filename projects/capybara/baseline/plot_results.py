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
import os
import pandas as pd

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

from baseline_utils import predictions_vote

from plot_utils import (plot_confusion_matrix, plot_train_history,
                        plot_classification_report, plot_predictions)

if __name__ == '__main__':
  

  
  # Path to CSV files (training history and predictions)
  parser = argparse.ArgumentParser()
  parser.add_argument('--vote_window', '-v', dest='vote_window',
                      type=int, default=11)
  parser.add_argument('--input_dir', '-i', dest='input_dir',
                      type=str, default='results')
  parser.add_argument('--output_dir', '-o', dest='output_dir',type=str,
                      default='plots')
  options = parser.parse_args()
  vote_window = options.vote_window
  input_dir = options.input_dir
  output_dir = options.output_dir
  
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  train_history_path = os.path.join(input_dir, 'train_history.csv')
  predictions_path = os.path.join(input_dir, 'predictions.csv')
  
  # Training history
  df = pd.read_csv(train_history_path)
  epochs = range(len(df.epoch.values))
  acc = df.acc.values
  loss = df.loss.values
  output_file = os.path.join(output_dir, 'train_history.html')
  plot_train_history(epochs, acc, loss, output_file)
  print 'Plot saved:', output_file

  # Predictions
  df = pd.read_csv(predictions_path)
  t = df.t.values
  X_values = df.scalar_value.values
  y_true = df.y_true.values
  y_pred = df.y_pred.values
  
  if vote_window > 0:
    y_pred = predictions_vote(y_pred, vote_window)

  # Accuracy
  acc = accuracy_score(y_true, y_pred)
  print 'Accuracy on test set:', acc
  
  label_list = sorted(df.y_true.unique())

  # Plot normalized confusion matrix
  cnf_matrix = confusion_matrix(y_true, y_pred)
  output_file = os.path.join(output_dir, 'confusion_matrix.png')
  _ = plot_confusion_matrix(cnf_matrix,
                            output_file,
                            classes=label_list,
                            normalize=True,
                            title='Confusion matrix (accuracy=%.2f)' % acc)
  print 'Plot saved:', output_file

  # Classification report (F1 score, etc.)
  clf_report = classification_report(y_true, y_pred)
  output_file = os.path.join(output_dir, 'classification_report.png')
  plot_classification_report(clf_report, output_file)
  print 'Plot saved:', output_file

  # Plot predictions
  output_file = os.path.join(output_dir, 'predictions.html')
  title = 'Predictions (accuracy=%s)' % acc
  plot_predictions(t, X_values, y_true, y_pred, output_file, title)
  print 'Plot saved:', output_file
