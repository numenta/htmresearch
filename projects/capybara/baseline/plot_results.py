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

from sklearn.metrics import classification_report, confusion_matrix
from plot_utils import (plot_confusion_matrix, plot_train_history,
                        plot_classification_report)

LABELS = {
  1: 'WALKING',
  2: 'WALKING_UPSTAIRS',
  3: 'WALKING_DOWNSTAIRS',
  4: ' SITTING',
  5: 'STANDING',
  6: 'LAYING'
}

OUTPUT_DIR = 'plots'

if __name__ == '__main__':
  
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  
  # Path to CSV files (training history and predictions)
  parser = argparse.ArgumentParser()
  parser.add_argument('--train-history', '-t', dest='train_history',type=str,
                      default='results/train_history.csv')
  parser.add_argument('--predictions', '-p', dest='predictions',type=str,
                      default='results/predictions.csv')
  options = parser.parse_args()
  train_history_path = options.train_history
  predictions_path = options.predictions
  
  # Training history
  df = pd.read_csv(train_history_path)
  epochs = range(len(df.epoch.values))
  acc = df.acc.values
  loss = df.loss.values
  output_file = os.path.join(OUTPUT_DIR, 'train_history.html')
  plot_train_history(epochs, acc, loss, output_file)

  # Predictions
  df = pd.read_csv(predictions_path)
  y_true = df.y_true.values
  y_pred = df.y_pred.values
  y_vote = df.y_vote.values

  # Find labels in use
  label_list = df.y_true.unique()
  label_list = [LABELS[l] for l in label_list]

  # Plot normalized confusion matrix
  cnf_matrix = confusion_matrix(y_true, y_pred)
  output_file = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
  _ = plot_confusion_matrix(cnf_matrix,
                            output_file,
                            classes=label_list,
                            normalize=True,
                            title='Confusion matrix')

  # Classification report (F1 score, etc.)
  clf_report = classification_report(y_true, y_pred, target_names=label_list)
  output_file = os.path.join(OUTPUT_DIR, 'classification_report.png')
  plot_classification_report(clf_report, output_file)
