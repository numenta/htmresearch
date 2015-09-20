#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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


from matplotlib import pyplot as plt
plt.ion()

from errorMetrics import *
import pandas as pd

from pylab import rcParams
from plot import ExperimentResult, plotAccuracy, computeSquareDeviation, computeLikelihood
import plotly.plotly as py

rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.facecolor': 'white'})
rcParams.update({'ytick.labelsize': 8})
rcParams.update({'figure.figsize': (12, 6)})

window = 480
figPath = './result/'

plt.close('all')

# use datetime as x-axis
dataSet = 'nyc_taxi'
filePath = './data/' + dataSet + '.csv'
data = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['datetime', 'value', 'timeofday', 'dayofweek'])

xaxis_datetime = pd.to_datetime(data['datetime'])

def plotLSTMresult(experiment, window, xaxis=None, label=None):
  expResult = ExperimentResult(experiment)

  if xaxis is not None:
    x = xaxis
  else:
    x = range(0, len(expResult.error))

  plotAccuracy((expResult.error, x),
               expResult.truth,
               train=expResult.train,
               window=window,
               label=label,
               params=expResult.params,
               errorType=expResult.errorType)


### Figure 1: Continuous vs Batch LSTM
fig = plt.figure(1)
plotLSTMresult('results/nyc_taxi_experiment_one_shot/',
               window, xaxis=xaxis_datetime, label='static lstm')
plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window5001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-5000')
plt.legend()
plt.savefig(figPath + 'continuousVsbatch.pdf')


### Figure 2: Continuous LSTM with different window size

fig = plt.figure(2)
plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window1001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-1000')

plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window3001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-3000')

plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window5001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-5000')
plt.legend()
plt.savefig(figPath + 'continuous.pdf')


### Figure 3: Continuous LSTM & TM on perturbed data
fig = plt.figure(3)
plotLSTMresult('results/nyc_taxi_experiment_perturb/learning_window1001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-1000')

plotLSTMresult('results/nyc_taxi_experiment_perturb/learning_window3001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-3000')

plotLSTMresult('results/nyc_taxi_experiment_perturb/learning_window5001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-5000')

# load TM prediction
filePath = './data/' + 'nyc_taxi' + '.csv'
data = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['datetime', 'value', 'timeofday', 'dayofweek'])

dataSet = 'nyc_taxi_perturb'
filePath = './prediction/' + dataSet + '_TM_pred.csv'
predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
truth = predData_TM['value']
predData_TM_five_step = np.roll(predData_TM['prediction5'], 5)
iteration = predData_TM.index

(square_deviation, x) = computeSquareDeviation(predData_TM_five_step, truth, iteration)
square_deviation[:5000] = None
x = pd.to_datetime(data['datetime'])
plotAccuracy((square_deviation, x),
             truth,
             window=window,
             label='TM')
plt.legend()
plt.savefig(figPath + 'continuous_perturb.pdf')


### Figure 4: Continuous LSTM with different window size

fig = plt.figure(4)
plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window1001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-1000')

plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window3001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-3000')

plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window5001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-5000')
plt.legend()
plt.savefig(figPath + 'continuous.pdf')

