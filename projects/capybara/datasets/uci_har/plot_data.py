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
cwd = os.getcwd()
input_dir = os.path.join(cwd, os.pardir, os.pardir,
                         'classification', 'data', 'uci')

LABELS = {
  1: 'WALKING',
  2: 'WALKING_UPSTAIRS',
  3: 'WALKING_DOWNSTAIRS',
  4: ' SITTING',
  5: 'STANDING',
  6: 'LAYING'
}

# range to plot
x_min = 0
x_max = 50000
metric = 'body_acc_x'
test_file = '%s_inertial_signals_test.csv' % metric
train_file = '%s_inertial_signals_train.csv' % metric

train = pd.read_csv(os.path.join(input_dir, train_file))[x_min:x_max]
test = pd.read_csv(os.path.join(input_dir, test_file))[x_min:x_max]

train = train.drop([0, 1])
test = test.drop([0, 1])

train.y = train.y.astype(float)
test.y = test.y.astype(float)
train.label = train.label.astype(int)
test.label = test.label.astype(int)

data = {
  'training': train,
  'test': test
}

filenames = []
for phase in data.keys():
  df = data[phase]
  min_val = min(df.y.values)
  max_val = max(df.y.values)

  traces = []
  traces.append(go.Scatter(x=df.x,
                           y=df.y,
                           name='Data'))
  layout = go.Layout(title='%s data (%s)' % (phase, metric))
  labels = [l for l in df.label.unique()]
  num_labels = len(labels)

  # make sure we have enough colors
  colors *= num_labels // 10 + 1

  for label in labels:
    lower_bound = df.copy()
    upper_bound = df.copy()

    lower_bound.y = min_val
    upper_bound.y[df.label == label] = max_val
    upper_bound.y[df.label != label] = min_val

    traces.append(go.Scatter(x=upper_bound.x,
                             y=upper_bound.y,
                             mode='lines',
                             fill=None,
                             line=dict(color=colors[label]),
                             legendgroup=str(label),
                             name='label %s' % label))

    traces.append(go.Scatter(x=lower_bound.x,
                             y=lower_bound.y,
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
