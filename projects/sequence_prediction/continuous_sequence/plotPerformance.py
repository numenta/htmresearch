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
from plot import ExperimentResult, plotAccuracy, computeSquareDeviation, computeLikelihood, plotLSTMresult

rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.facecolor': 'white'})
rcParams.update({'ytick.labelsize': 8})
rcParams.update({'figure.figsize': (12, 6)})
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

window = 960
figPath = './result/'

plt.close('all')

# use datetime as x-axis
dataSet = 'nyc_taxi'
filePath = './data/' + dataSet + '.csv'
data = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['datetime', 'value', 'timeofday', 'dayofweek'])

xaxis_datetime = pd.to_datetime(data['datetime'])


def computeAltMAPE(truth, prediction, startFrom=0):
  return np.nanmean(np.abs(truth[startFrom:] - prediction[startFrom:]))/np.nanmean(np.abs(truth[startFrom:]))

expResult = ExperimentResult('results/nyc_taxi_experiment_continuous/learning_window6001.0/')

### Figure 1: Continuous vs Batch LSTM
fig = plt.figure()
# NRMSE_StaticLSTM = plotLSTMresult('results/nyc_taxi_experiment_one_shot/',
#                                   window, xaxis=xaxis_datetime, label='static lstm')
(NRMSE_LSTM6000, expResult_LSTM6000) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window6001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-6000')
plt.legend()
plt.savefig(figPath + 'continuousVsbatch.pdf')


### Figure 2: Continuous LSTM with different window size

fig = plt.figure()
(NRMSE_LSTM1000, expResult_LSTM1000) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window1001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-1000')

(NRMSE_LSTM3000, expResult_LSTM3000) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window3001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-3000')

(NRMSE_LSTM6000, expResult_LSTM6000) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window6001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-6000')

dataSet = 'nyc_taxi'
filePath = './prediction/' + dataSet + '_TM_pred.csv'
predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
tm_truth = np.roll(predData_TM['value'], -5)
predData_TM_five_step = np.array(predData_TM['prediction5'])

square_deviation = computeSquareDeviation(predData_TM_five_step, tm_truth)
square_deviation[:6000] = None

NRMSE_TM = plotAccuracy((square_deviation, xaxis_datetime),
                       tm_truth,
                       window=window,
                       errorType='square_deviation',
                       label='TM')


filePath = './prediction/' + dataSet + '_ESN_pred.csv'
predData_ESN = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
esn_truth = np.roll(predData_ESN['value'], -5)
predData_ESN_five_step = np.array(predData_ESN['prediction5'])

square_deviation = computeSquareDeviation(predData_ESN_five_step, esn_truth)
square_deviation[:6000] = None

NRMSE_ESN = plotAccuracy((square_deviation, xaxis_datetime),
                       tm_truth,
                       window=window,
                       errorType='square_deviation',
                       label='ESN')


filePath = './prediction/' + dataSet + '_plainKNN_pred.csv'
predData_KNN = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
knn_truth = np.roll(predData_KNN['value'], -5)
predData_KNN_five_step = np.array(predData_KNN['prediction5'])

square_deviation = computeSquareDeviation(predData_KNN_five_step, knn_truth)
square_deviation[:6000] = None

NRMSE_KNN = plotAccuracy((square_deviation, xaxis_datetime),
                       tm_truth,
                       window=window,
                       errorType='square_deviation',
                       label='KNN')

filePath = './prediction/' + dataSet + '_kNN2_pred.csv'
predData_KNN2 = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
knn2_truth = np.roll(predData_KNN2['value'], -5)
predData_KNN2_five_step = np.array(predData_KNN2['prediction5'])

square_deviation = computeSquareDeviation(predData_KNN2_five_step, knn_truth)
square_deviation[:6000] = None

NRMSE_KNN2 = plotAccuracy((square_deviation, xaxis_datetime),
                       tm_truth,
                       window=window,
                       errorType='square_deviation',
                       label='KNN')

filePath = './prediction/' + dataSet + '_ARIMA_pred.csv'
predData_ARIMA = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                             names=['step', 'value', 'prediction1', 'prediction5'])
arima_truth = np.roll(predData_ARIMA['value'], -5)
predData_ARIMA_five_step = predData_ARIMA['prediction5']

square_deviation = computeSquareDeviation(predData_ARIMA_five_step, arima_truth)
square_deviation[:6000] = None
NRMSE_ARIMA = plotAccuracy((square_deviation, xaxis_datetime),
                       arima_truth,
                       window=window,
                       errorType='square_deviation',
                       label='ARIMA')


predData_shift_five_step = np.roll(tm_truth, 5).astype('float32')
square_deviation = computeSquareDeviation(predData_shift_five_step, tm_truth)
square_deviation[:6000] = None
NRMSE_Shift = plotAccuracy((square_deviation, xaxis_datetime),
                       tm_truth,
                       window=window,
                       errorType='square_deviation',
                       label='Shift')

plt.legend()
plt.savefig(figPath + 'continuous.pdf')


### Figure 3: Continuous LSTM with different window size using the likelihood metric

fig = plt.figure()
# negLL_StaticLSTM = \
#   plotLSTMresult('results/nyc_taxi_experiment_one_shot_likelihood/',
#                window, xaxis=xaxis_datetime, label='static LSTM ')

plt.clf()
(negLL_LSTM1000, expResult_LSTM1000_negLL) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window1001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-1000')

(negLL_LSTM3000, expResult_LSTM3000_negLL) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window3001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-3000')

(negLL_LSTM6000, expResult_LSTM6000_negLL) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window6001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-6000')

dataSet = 'nyc_taxi'
tm_prediction = np.load('./result/'+dataSet+'TMprediction.npy')
tm_truth = np.load('./result/'+dataSet+'TMtruth.npy')
from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
negLL = computeLikelihood(tm_prediction, tm_truth, encoder)
negLL[:6000] = None
negLL_TM = \
  plotAccuracy((negLL, xaxis_datetime), tm_truth, window=window, errorType='negLL', label='TM')
plt.legend()
plt.savefig(figPath + 'continuous_likelihood.pdf')


### Figure 4: Continuous LSTM with different window size using the likelihood metric

fig = plt.figure()
# negLL_StaticLSTM = \
#   plotLSTMresult('results/nyc_taxi_experiment_one_shot_likelihood/',
#                window, xaxis=xaxis_datetime, label='static LSTM ')

plt.clf()
(negLL_LSTM1000, expResult_LSTM1000_negLL) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window1001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-1000')

(negLL_LSTM3000, expResult_LSTM3000_negLL) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window3001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-3000')

(negLL_LSTM6000, expResult_LSTM6000_negLL) = \
  plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window6001.0/',
               window, xaxis=xaxis_datetime, label='continuous LSTM-6000')

dataSet = 'nyc_taxi'
tm_prediction = np.load('./result/'+dataSet+'TMprediction.npy')
tm_truth = np.load('./result/'+dataSet+'TMtruth.npy')
from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
negLL = computeLikelihood(tm_prediction, tm_truth, encoder)
negLL[:6000] = None
negLL_TM = \
  plotAccuracy((negLL, xaxis_datetime), tm_truth, window=window, errorType='negLL', label='TM')
plt.legend()
plt.savefig(figPath + 'continuous_likelihood.pdf')


startFrom = 6000
altMAPE_LSTM6000 = computeAltMAPE(expResult_LSTM6000.truth, expResult_LSTM6000.predictions, startFrom)
altMAPE_LSTM3000 = computeAltMAPE(expResult_LSTM3000.truth, expResult_LSTM3000.predictions, startFrom)
altMAPE_LSTM1000 = computeAltMAPE(expResult_LSTM1000.truth, expResult_LSTM1000.predictions, startFrom)
altMAPE_TM = computeAltMAPE(tm_truth, predData_TM_five_step, startFrom)
altMAPE_ARIMA = computeAltMAPE(arima_truth, predData_ARIMA_five_step, startFrom)
altMAPE_ESN = computeAltMAPE(esn_truth, predData_ESN_five_step, startFrom)
altMAPE_KNN = computeAltMAPE(knn_truth, predData_KNN_five_step, startFrom)
altMAPE_KNN2 = computeAltMAPE(knn2_truth, predData_KNN2_five_step, startFrom)
altMAPE_Shift = computeAltMAPE(tm_truth, predData_shift_five_step, startFrom)


truth = tm_truth
NRMSE_Shift_mean = np.sqrt(np.nanmean(NRMSE_Shift))/np.nanstd(truth)
NRMSE_ARIMA_mean = np.sqrt(np.nanmean(NRMSE_ARIMA))/np.nanstd(truth)
NRMSE_ESN_mean = np.sqrt(np.nanmean(NRMSE_ESN))/np.nanstd(truth)
NRMSE_KNN_mean = np.sqrt(np.nanmean(NRMSE_KNN))/np.nanstd(truth)
NRMSE_KNN2_mean = np.sqrt(np.nanmean(NRMSE_KNN2))/np.nanstd(truth)
NRMSE_TM_mean = np.sqrt(np.nanmean(NRMSE_TM))/np.nanstd(truth)
NRMSE_LSTM1000_mean = np.sqrt(np.nanmean(NRMSE_LSTM1000))/np.nanstd(truth)
NRMSE_LSTM3000_mean = np.sqrt(np.nanmean(NRMSE_LSTM3000))/np.nanstd(truth)
NRMSE_LSTM6000_mean = np.sqrt(np.nanmean(NRMSE_LSTM6000))/np.nanstd(truth)


# fig, ax = plt.subplots(nrows=1, ncols=1)
# inds = np.arange(6)
# ax.bar(inds, [NRMSE_Shift_mean,
#                  NRMSE_ARIMA_mean,
#                  NRMSE_TM_mean,
#                  NRMSE_LSTM1000_mean,
#                  NRMSE_LSTM3000_mean,
#                  NRMSE_LSTM6000_mean], width=0.2)
# ax.set_xticks(inds+0.3/2)
# ax.set_xticklabels( ('Shift', 'ARIMA', 'TM', 'LSTM1000', 'LSTM3000', 'LSTM6000') )
# # Make the y-axis label and tick labels match the line color.
# ax.set_ylabel('NRMSE', color='b')
# for tl in ax.get_yticklabels():
#     tl.set_color('b')
#
# inds = np.arange(6)
# ax2 = ax.twinx()
# ax2.set_ylabel('negative Log-likelihood', color='r')
# for tl in ax2.get_yticklabels():
#     tl.set_color('r')
#
# ax2.bar(inds+0.3, [0, 0,
#                np.nanmean(negLL_TM),
#                np.nanmean(negLL_LSTM1000),
#                np.nanmean(negLL_LSTM3000),
#                np.nanmean(negLL_LSTM6000)], width=0.2, color='r')
# plt.savefig(figPath  + 'model_performance_summary.pdf')


fig, ax = plt.subplots(nrows=1, ncols=3)
inds = np.arange(9)
ax1 = ax[0]
width = 0.5
ax1.bar(inds, [NRMSE_Shift_mean,
                 NRMSE_ARIMA_mean,
                 NRMSE_KNN_mean,
                 NRMSE_KNN2_mean,
                 NRMSE_ESN_mean,
                 NRMSE_LSTM1000_mean,
                 NRMSE_LSTM3000_mean,
                 NRMSE_LSTM6000_mean,
                 NRMSE_TM_mean], width=width)
ax1.set_xticks(inds+width/2)
ax1.set_ylabel('NRMSE')
ax1.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
ax1.set_xticklabels( ('Shift', 'ARIMA', 'KNN', 'KNN2', 'ESN',
                      'LSTM1000', 'LSTM3000', 'LSTM6000', 'TM') )
for tick in ax1.xaxis.get_major_ticks():
  tick.label.set_rotation('vertical')

ax3 = ax[1]
ax3.bar(inds, [altMAPE_Shift,
               altMAPE_ARIMA,
               altMAPE_KNN,
               altMAPE_KNN2,
               altMAPE_ESN,
               altMAPE_LSTM1000,
               altMAPE_LSTM3000,
               altMAPE_LSTM6000,
               altMAPE_TM], width=width, color='b')
ax3.set_xticks(inds+width/2)
ax3.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
ax3.set_ylabel('MAPE')
ax3.set_xticklabels( ('Shift', 'ARIMA', 'KNN', 'KNN2', 'ESN',
                      'LSTM1000', 'LSTM3000', 'LSTM6000', 'TM') )
for tick in ax3.xaxis.get_major_ticks():
  tick.label.set_rotation('vertical')

ax2 = ax[2]
ax2.set_ylabel('Negative Log-likelihood')
ax2.bar(inds, [np.nanmean(negLL_LSTM1000),
               np.nanmean(negLL_LSTM3000),
               np.nanmean(negLL_LSTM6000),
               np.nanmean(negLL_TM), 0, 0, 0, 0, 0], width=width, color='b')
ax2.set_xticks(inds+width/2)
ax2.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
ax2.set_xticklabels(('LSTM1000', 'LSTM3000', 'LSTM6000', 'TM', '', '', ''))
for tick in ax2.xaxis.get_major_ticks():
  tick.label.set_rotation('vertical')

plt.savefig(figPath + 'model_performance_summary_alternative.pdf')


### Figure 6:
# fig = plt.figure(6)
# plt.plot(xaxis_datetime, tm_truth, label='Before')
# plt.plot(xaxis_datetime, tm_truth_perturb, label='After')
# plt.xlim([xaxis_datetime[13050], xaxis_datetime[13480]])
# plt.ylabel('30min Passenger Count')
# plt.legend()
# plt.savefig(figPath + 'example_perturbed_data.pdf')


### Plot Example Data Segments
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
fig, ax = plt.subplots()

ax.plot(xaxis_datetime, tm_truth, 'k-o')
ax.xaxis.set_major_locator( DayLocator() )
ax.xaxis.set_minor_locator( HourLocator(range(0,25,6)) )
ax.xaxis.set_major_formatter( DateFormatter('%Y-%m-%d') )
ax.set_xlim([xaxis_datetime[14060], xaxis_datetime[14400]])
yticklabel = ax.get_yticks()/1000
new_yticklabel = []
for i in range(len(yticklabel)):
  new_yticklabel.append( str(int(yticklabel[i]))+' k')
ax.set_yticklabels(new_yticklabel)
ax.set_ylim([0, 30000])
ax.set_ylabel('Passenger Count in 30 min window')
plt.savefig(figPath + 'example_data.pdf')