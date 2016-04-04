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
from errorMetrics import *
import pandas as pd

from pylab import rcParams
from plot import ExperimentResult, plotAccuracy, computeSquareDeviation, computeLikelihood, plotLSTMresult
from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder

rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.facecolor': 'white'})
rcParams.update({'ytick.labelsize': 8})
rcParams.update({'figure.figsize': (12, 6)})
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')

window = 960
skipTrain = 6000
figPath = './result/'

def getDatetimeAxis():
  """
  use datetime as x-axis
  """
  dataSet = 'nyc_taxi'
  filePath = './data/' + dataSet + '.csv'
  data = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                     names=['datetime', 'value', 'timeofday', 'dayofweek'])

  xaxisDate = pd.to_datetime(data['datetime'])
  return xaxisDate


def computeAltMAPE(truth, prediction, startFrom=0):
  return np.nanmean(np.abs(truth[startFrom:] - prediction[startFrom:]))/np.nanmean(np.abs(truth[startFrom:]))


def computeNRMSE(truth, prediction, startFrom=0):
  squareDeviation = computeSquareDeviation(prediction, truth)
  squareDeviation[:startFrom] = None
  return np.sqrt(np.nanmean(squareDeviation))/np.nanstd(truth)


def loadExperimentResult(filePath):
  expResult = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                            names=['step', 'value', 'prediction5'])
  groundTruth = np.roll(expResult['value'], -5)
  prediction5step = np.array(expResult['prediction5'])
  return (groundTruth, prediction5step)



if __name__ == "__main__":
  xaxisDate = getDatetimeAxis()
  expResult = ExperimentResult('results/nyc_taxi_experiment_continuous/learning_window6001.0/')

  # ### Figure 1: Continuous vs Batch LSTM
  # fig = plt.figure()
  # # NRMSE_StaticLSTM = plotLSTMresult('results/nyc_taxi_experiment_one_shot/',
  # #                                   window, xaxis=xaxis_datetime, label='static lstm')
  # (nrmseLSTM6000, expResultLSTM6000) = \
  #   plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window6001.0/',
  #                  window, xaxis=xaxisDate, label='continuous LSTM-6000')
  # plt.legend()
  # plt.savefig(figPath + 'continuousVsbatch.pdf')
  #

  ### Figure 2: Continuous LSTM with different window size

  fig = plt.figure()
  (nrmseLSTM1000, expResultLSTM1000) = \
    plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window1001.0/',
                   window, xaxis=xaxisDate, label='continuous LSTM-1000')

  (nrmseLSTM3000, expResultLSTM3000) = \
    plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window3001.0/',
                   window, xaxis=xaxisDate, label='continuous LSTM-3000')

  (nrmseLSTM6000, expResultLSTM6000) = \
    plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window6001.0/',
                   window, xaxis=xaxisDate, label='continuous LSTM-6000')

  dataSet = 'nyc_taxi'
  filePath = './prediction/' + dataSet + '_TM_pred.csv'

  (tmTruth, tmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_TM_pred.csv')

  squareDeviation = computeSquareDeviation(tmPrediction, tmTruth)
  squareDeviation[:skipTrain] = None

  nrmseTM = plotAccuracy((squareDeviation, xaxisDate),
                         tmTruth,
                         window=window,
                         errorType='square_deviation',
                         label='TM')


  (esnTruth, esnPrediction) = loadExperimentResult('./prediction/' + dataSet + '_ESN_pred.csv')

  squareDeviation = computeSquareDeviation(esnPrediction, esnTruth)
  squareDeviation[:skipTrain] = None

  nrmseESN = plotAccuracy((squareDeviation, xaxisDate),
                          tmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='ESN')


  (knnTruth, knnPrediction) = loadExperimentResult('./prediction/' + dataSet + '_plainKNN_pred.csv')

  squareDeviation = computeSquareDeviation(knnPrediction, knnTruth)
  squareDeviation[:skipTrain] = None

  nrmseKNN = plotAccuracy((squareDeviation, xaxisDate),
                          tmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='KNN')


  (arimaTruth, arimaPrediction) = loadExperimentResult('./prediction/' + dataSet + '_ARIMA_pred.csv')

  squareDeviation = computeSquareDeviation(arimaPrediction, arimaTruth)
  squareDeviation[:skipTrain] = None
  nrmseARIMA = plotAccuracy((squareDeviation, xaxisDate),
                            arimaTruth,
                            window=window,
                            errorType='square_deviation',
                            label='ARIMA')


  (adaptiveFilterTruth, adaptiveFilterPrediction) = loadExperimentResult('./prediction/' + dataSet + '_adaptiveFilter_pred.csv')
  squareDeviation = computeSquareDeviation(adaptiveFilterPrediction, adaptiveFilterTruth)
  squareDeviation[:skipTrain] = None
  nrmseAdaptiveFilter = plotAccuracy((squareDeviation, xaxisDate),
                                     adaptiveFilterTruth,
                                     window=window,
                                     errorType='square_deviation',
                                     label='AdaptiveFilter')


  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_elm_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='Extreme Learning Machine')


  shiftPrediction = np.roll(tmTruth, 5).astype('float32')
  squareDeviation = computeSquareDeviation(shiftPrediction, tmTruth)
  squareDeviation[:skipTrain] = None
  nrmseShift = plotAccuracy((squareDeviation, xaxisDate),
                            tmTruth,
                            window=window,
                            errorType='square_deviation',
                            label='Shift')

  plt.legend()
  plt.savefig(figPath + 'continuous.pdf')


  ### Figure 3: Continuous LSTM with different window size using the likelihood metric

  # fig = plt.figure()
  # # negLL_StaticLSTM = \
  # #   plotLSTMresult('results/nyc_taxi_experiment_one_shot_likelihood/',
  # #                window, xaxis=xaxis_datetime, label='static LSTM ')
  #
  # plt.clf()
  # (negLLLSTM1000, expResultLSTM1000negLL) = \
  #   plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window1001.0/',
  #                  window, xaxis=xaxisDate, label='continuous LSTM-1000')
  #
  # (negLLLSTM3000, expResultLSTM3000negLL) = \
  #   plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window3001.0/',
  #                  window, xaxis=xaxisDate, label='continuous LSTM-3000')
  #
  # (negLLLSTM6000, expResultLSTM6000negLL) = \
  #   plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window6001.0/',
  #                  window, xaxis=xaxisDate, label='continuous LSTM-6000')
  #
  # dataSet = 'nyc_taxi'
  # tm_prediction = np.load('./result/'+dataSet+'TMprediction.npy')
  # tmTruth = np.load('./result/' + dataSet + 'TMtruth.npy')
  #
  # encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
  # negLL = computeLikelihood(tm_prediction, tmTruth, encoder)
  # negLL[:skipTrain] = None
  # negLLTM = plotAccuracy((negLL, xaxisDate), tmTruth,
  #                        window=window, errorType='negLL', label='TM')
  # plt.legend()
  # plt.savefig(figPath + 'continuous_likelihood.pdf')


  ### Figure 4: Continuous LSTM with different window size using the likelihood metric

  fig = plt.figure()
  plt.clf()
  (negLLLSTM1000, expResultLSTM1000negLL) = \
    plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window1001.0/',
                   window, xaxis=xaxisDate, label='continuous LSTM-1000')

  (negLLLSTM3000, expResultLSTM3000negLL) = \
    plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window3001.0/',
                   window, xaxis=xaxisDate, label='continuous LSTM-3000')

  (negLLLSTM6000, expResultLSTM6000negLL) = \
    plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window6001.0/',
                   window, xaxis=xaxisDate, label='continuous LSTM-6000')

  dataSet = 'nyc_taxi'
  tmPredictionLL = np.load('./result/'+dataSet+'TMprediction.npy')
  tmTruth = np.load('./result/' + dataSet + 'TMtruth.npy')

  encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
  negLL = computeLikelihood(tmPredictionLL, tmTruth, encoder)
  negLL[:skipTrain] = None
  negLLTM = plotAccuracy((negLL, xaxisDate), tmTruth,
                         window=window, errorType='negLL', label='TM')
  plt.legend()
  plt.savefig(figPath + 'continuous_likelihood.pdf')


  startFrom = skipTrain
  altMAPELSTM6000 = computeAltMAPE(expResultLSTM6000.truth, expResultLSTM6000.predictions, startFrom)
  altMAPELSTM3000 = computeAltMAPE(expResultLSTM3000.truth, expResultLSTM3000.predictions, startFrom)
  altMAPELSTM1000 = computeAltMAPE(expResultLSTM1000.truth, expResultLSTM1000.predictions, startFrom)
  altMAPETM = computeAltMAPE(tmTruth, tmPrediction, startFrom)
  altMAPEARIMA = computeAltMAPE(arimaTruth, arimaPrediction, startFrom)
  altMAPEESN = computeAltMAPE(esnTruth, esnPrediction, startFrom)
  altMAPEKNN = computeAltMAPE(knnTruth, knnPrediction, startFrom)
  altMAPEShift = computeAltMAPE(tmTruth, shiftPrediction, startFrom)
  altMAPEAdaptiveFilter = computeAltMAPE(tmTruth, adaptiveFilterPrediction, startFrom)
  altMAPEELM = computeAltMAPE(elmTruth, elmPrediction, startFrom)

  truth = tmTruth
  nrmseShiftMean = np.sqrt(np.nanmean(nrmseShift)) / np.nanstd(truth)
  nrmseARIMAmean = np.sqrt(np.nanmean(nrmseARIMA)) / np.nanstd(truth)
  nrmseESNmean = np.sqrt(np.nanmean(nrmseESN)) / np.nanstd(truth)
  nrmseKNNmean = np.sqrt(np.nanmean(nrmseKNN)) / np.nanstd(truth)
  nrmseTMmean = np.sqrt(np.nanmean(nrmseTM)) / np.nanstd(truth)
  nrmseELMmean = np.sqrt(np.nanmean(nrmseELM)) / np.nanstd(truth)
  nrmseLSTM1000mean = np.sqrt(np.nanmean(nrmseLSTM1000)) / np.nanstd(truth)
  nrmseLSTM3000mean = np.sqrt(np.nanmean(nrmseLSTM3000)) / np.nanstd(truth)
  nrmseLSTM6000mean = np.sqrt(np.nanmean(nrmseLSTM6000)) / np.nanstd(truth)


  fig, ax = plt.subplots(nrows=1, ncols=3)
  inds = np.arange(7)
  ax1 = ax[0]
  width = 0.5
  ax1.bar(inds, [nrmseARIMAmean,
                 nrmseELMmean,
                 nrmseESNmean,
                 nrmseLSTM1000mean,
                 nrmseLSTM3000mean,
                 nrmseLSTM6000mean,
                 nrmseTMmean], width=width)
  ax1.set_xticks(inds+width/2)
  ax1.set_ylabel('NRMSE')
  ax1.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax1.set_xticklabels( ('ARIMA', 'ELM',  'ESN',
                        'LSTM1000', 'LSTM3000', 'LSTM6000', 'HTM') )
  for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')

  ax3 = ax[1]
  ax3.bar(inds, [altMAPEARIMA,
                 altMAPEELM,
                 altMAPEESN,
                 altMAPELSTM1000,
                 altMAPELSTM3000,
                 altMAPELSTM6000,
                 altMAPETM], width=width, color='b')
  ax3.set_xticks(inds+width/2)
  ax3.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax3.set_ylabel('MAPE')
  ax3.set_xticklabels( ('ARIMA', 'ELM', 'ESN',
                        'LSTM1000', 'LSTM3000', 'LSTM6000', 'HTM') )
  for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')

  ax2 = ax[2]
  ax2.set_ylabel('Negative Log-likelihood')
  ax2.bar(inds, [np.nanmean(negLLLSTM1000),
                 np.nanmean(negLLLSTM3000),
                 np.nanmean(negLLLSTM6000),
                 np.nanmean(negLLTM), 0, 0, 0], width=width, color='b')
  ax2.set_xticks(inds+width/2)
  ax2.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax2.set_ylim([0, 2.0])
  ax2.set_xticklabels(('LSTM1000', 'LSTM3000', 'LSTM6000', 'HTM', '', '', ''))
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

  ax.plot(xaxisDate, tmTruth, 'k-o')
  ax.xaxis.set_major_locator( DayLocator() )
  ax.xaxis.set_minor_locator( HourLocator(range(0,25,6)) )
  ax.xaxis.set_major_formatter( DateFormatter('%Y-%m-%d') )
  ax.set_xlim([xaxisDate[14060], xaxisDate[14400]])
  yticklabel = ax.get_yticks()/1000
  new_yticklabel = []
  for i in range(len(yticklabel)):
    new_yticklabel.append( str(int(yticklabel[i]))+' k')
  ax.set_yticklabels(new_yticklabel)
  ax.set_ylim([0, 30000])
  ax.set_ylabel('Passenger Count in 30 min window')
  plt.savefig(figPath + 'example_data.pdf')