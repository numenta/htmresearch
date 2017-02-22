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

import os
import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go
import colorlover as cl

colors = cl.scales['10']['qual']['Set3']

LABELS = ['WALKING',
          'WALKING_UPSTAIRS',
          'WALKING_DOWNSTAIRS',
          'SITTING',
          'STANDING',
          'LAYING']

# Range to plot
x_min = 0
x_max = 40000
metric = 'body_acc_x'

# Input data to plot
test_file = 'inertial_signals_test.csv'
train_file = 'inertial_signals_train.csv'
train = pd.read_csv(train_file)[x_min:x_max]
test = pd.read_csv(test_file)[x_min:x_max]

train[metric] = train[metric].astype(float)
test[metric] = test[metric].astype(float)
train.label = train.label.astype(int)
test.label = test.label.astype(int)

data = {
  'train': train,
  'test': test
}

filenames = []
for phase in data.keys():
  df = data[phase]
  min_val = min(df[metric].values)
  max_val = max(df[metric].values)

  traces = []
  traces.append(go.Scatter(x=range(len(df)),
                           y=df[metric],
                           name='Data'))
  layout = go.Layout(title='%s data (%s)' % (phase, metric))
  labels = [l for l in df.label.unique()]
  num_labels = len(labels)

  # make sure we have enough colors
  colors *= num_labels // 10 + 1

  for label in labels:
    lower_bound = df.copy()
    upper_bound = df.copy()

    lower_bound[metric] = min_val
    upper_bound[metric][df.label == label] = max_val
    upper_bound[metric][df.label != label] = min_val

    traces.append(go.Scatter(x=range(len(df)),
                             y=upper_bound[metric],
                             mode='lines',
                             fill=None,
                             line=dict(color=colors[label]),
                             legendgroup=str(label),
                             name='label %s' % label))

    traces.append(go.Scatter(x=range(len(df)),
                             y=lower_bound[metric],
                             mode='lines',
                             fill='tonexty',
                             legendgroup=str(label),
                             line=dict(color=colors[label]),
                             name='label %s (%s)' % (label, LABELS[label])))

  fig = go.Figure(data=traces, layout=layout)
  filename = '%s_%s.html' % (metric, phase)
  filenames.append(filename)
  py.plot(fig, filename=filename,
          auto_open=False)

print 'HTML plots saved: %s' % filenames
