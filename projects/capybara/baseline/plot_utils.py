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
import itertools

import plotly.offline as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt
import numpy as np



def show_values(pc, fmt="%.2f", **kw):
  """
  Heatmap with text in each cell with matplotlib's pyplot
  Source: http://stackoverflow.com/a/25074150/395857 
  By HYRY
  """
  from itertools import izip
  pc.update_scalarmappable()
  ax = pc.axes
  for p, color, value in izip(pc.get_paths(), pc.get_facecolors(),
                              pc.get_array()):
    x, y = p.vertices[:-2, :].mean(0)
    if np.all(color[:3] > 0.5):
      color = (0.0, 0.0, 0.0)
    else:
      color = (1.0, 1.0, 1.0)
    ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)



def cm2inch(*tupl):
  """
  Specify figure size in centimeter in matplotlib
  Source: http://stackoverflow.com/a/22787457/395857
  By gns-ank
  """
  inch = 2.54
  if type(tupl[0]) == tuple:
    return tuple(i / inch for i in tupl[0])
  else:
    return tuple(i / inch for i in tupl)



def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels,
            figure_width=40, figure_height=20, correct_orientation=False,
            cmap='RdBu'):
  """
  Inspired by:
  - http://stackoverflow.com/a/16124677/395857 
  - http://stackoverflow.com/a/25074150/395857
  """

  # Plot it out
  fig, ax = plt.subplots()
  # c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', 
  # linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
  c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2,
                cmap=cmap)

  # put the major ticks at the middle of each cell
  ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
  ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

  # set tick labels
  # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
  ax.set_xticklabels(xticklabels, minor=False)
  ax.set_yticklabels(yticklabels, minor=False)

  # set title and x/y labels
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  # Remove last blank column
  plt.xlim((0, AUC.shape[1]))

  # Turn off all the ticks
  ax = plt.gca()
  for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
  for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

  # Add color bar
  plt.colorbar(c)

  # Add text in each cell 
  show_values(c)

  # Proper orientation (origin at the top left instead of bottom left)
  if correct_orientation:
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # resize 
  fig = plt.gcf()
  # fig.set_size_inches(cm2inch(40, 20))
  # fig.set_size_inches(cm2inch(40*4, 20*4))
  fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, filename,
                               title='Classification report ', cmap='RdBu'):
  """
  Plot scikit-learn classification report.
  Extension based on http://stackoverflow.com/a/31689645/395857 
  """
  lines = classification_report.split('\n')

  classes = []
  plotMat = []
  support = []
  class_names = []
  for line in lines[2: (len(lines) - 2)]:
    t = line.strip().split()
    if len(t) < 2: continue
    classes.append(t[0])
    v = [float(x) for x in t[1: len(t) - 1]]
    support.append(int(t[-1]))
    class_names.append(t[0])
    plotMat.append(v)

  xlabel = 'Metrics'
  ylabel = 'Classes'
  xticklabels = ['Precision', 'Recall', 'F1-score']
  yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in
                 enumerate(support)]
  figure_width = 25
  figure_height = len(class_names) + 7
  correct_orientation = False
  heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels,
          figure_width, figure_height, correct_orientation, cmap=cmap)
  plt.savefig(filename,
              dpi=200,
              format='png',
              bbox_inches='tight')
  plt.close()



def plot_train_history(epochs, loss, acc, output_file):
  trace0 = go.Scatter(x=epochs, y=loss, name='Loss')
  trace1 = go.Scatter(x=epochs, y=acc, name='Accuracy')

  layout = go.Layout(showlegend=True, title='Training history')
  fig = go.Figure(data=[trace0, trace1], layout=layout)

  py.plot(fig,
          filename=output_file,
          auto_open=False,
          link_text=False)



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



def plot_predictions(t, X_values, y_true, y_pred, output_file_path):
  """
  Plot prediction results (correct and incorrect)  
  
  :param t: (list) timesteps
  :param X_values: (list) input scalar values (before any encoding)
  :param y_true: (list) true labels
  :param y_pred: (list) predicted labels
  :param output_file_path: (str) path to output file
  """
  if type(t) != np.ndarray:
    t = np.array(t)
  if type(X_values) != np.ndarray:
    X_values = np.array(X_values)
  if type(y_true) != np.ndarray:
    y_true = np.array(y_true)
  if type(y_pred) != np.ndarray:
    y_pred = np.array(y_pred)

  correct = []
  incorrect = []
  for r in y_true == y_pred:
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

  layout = go.Layout(showlegend=True, title='Predictions')
  fig = go.Figure(data=[trace0, trace1], layout=layout)

  py.plot(fig,
          filename=output_file_path,
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
  
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  
  # Use only 2 decimal numbers
  cm = np.around(cm, 2)

  plt.figure()
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=30)
  plt.yticks(tick_marks, classes)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(filename)
