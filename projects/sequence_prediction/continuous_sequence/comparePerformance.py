# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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
import csv, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from optparse import OptionParser
from swarm_runner import SwarmRunner
from errorMetrics import *
rcParams.update({'figure.autolayout': True})
plt.ion()
plt.close('all')


def loadTrueDataFile(filePath):
  data = pd.read_csv(filePath, header=0, nrows=1)
  colnames = data.columns.values
  ncol = len(colnames)

  # data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'value', 'prediction'])

  if ncol == 2:
    data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'value'])
  elif ncol == 3:
    data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'value', 'prediction'])
  elif ncol == 4:
    data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'value', 'timeofday', 'dayofweek'])
  return data



def loadDataFileLSTM(filePath):
  # data = pd.read_csv(filePath, header=0, nrows=1)
  # colnames = data.columns.values
  data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'value', 'prediction5'])

  return data


def loadDataFile(filePath):
  data = pd.read_csv(filePath, header=0, nrows=1)
  colnames = data.columns.values
  ncol = len(colnames)

  # data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'value', 'prediction'])

  if ncol == 2:
    data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'value'])
  elif ncol == 3:
    data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'value', 'prediction'])
  elif ncol == 4:
    data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'data', 'prediction1', 'prediction5'])
  return data


def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nCompare TM performance with trivial predictor using "
                              "model outputs in prediction directory "
                              "and outputting results to result directory.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default=0,
                    dest="dataSet",
                    help="DataSet Name, choose from sine, SantaFe_A, MackeyGlass")
  (options, args) = parser.parse_args(sys.argv[1:])

  return options, args


if __name__ == "__main__":

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  SWARM_CONFIG = SwarmRunner.importSwarmDescription(dataSet)
  nTrain = SWARM_CONFIG["streamDef"]['streams'][0]['last_record']
  print 'Compare Model performance for ', dataSet


  filePath = './data/' + dataSet + '.csv'
  print "load test data from ", filePath
  trueData = loadTrueDataFile(filePath)
  time_step = trueData['step']
  trueData = trueData['value'].astype('float')

  filePath = './prediction/' + dataSet + '_TM_pred.csv'
  print "load TM prediction from ", filePath
  predData_TM = loadDataFileLSTM(filePath)

  filePath = './prediction/' + dataSet + '_ARIMA_pred.csv'
  predData_ARIMA = loadDataFile(filePath)

  useTimeOfDay = True
  useDayOfWeek = True
  filePath = './prediction/'+dataSet+'_lstm_pred_useTimeOfDay_'+str(useTimeOfDay)+\
                   '_useDayOfWeek'+str(useDayOfWeek)+'.csv'
  predData_LSTM = loadDataFileLSTM(filePath)

  useTimeOfDay = True
  useDayOfWeek = False
  filePath = './prediction/'+dataSet+'_lstm_pred_useTimeOfDay_'+str(useTimeOfDay)+\
                   '_useDayOfWeek'+str(useDayOfWeek)+'.csv'
  predData_LSTM2 = loadDataFileLSTM(filePath)


  nTest = len(trueData) - nTrain - 5
  print "nTrain: ", nTrain
  print "nTest: ", len(trueData[nTrain:])

  # trivial shift predictor
  predData_shift = np.roll(trueData, 5)
  predData_TM_five_step = np.roll(predData_TM['prediction5'], 5)
  predData_ARIMA_five_step = np.roll(predData_ARIMA['prediction5'], 5)
  predData_LSTM_five_step = np.roll(predData_LSTM['prediction5'], 5)
  predData_LSTM_five_step2 = np.roll(predData_LSTM2['prediction5'], 5)


  # trueData = trueData[nTrain:nTrain+nTest]
  # predData_TM_five_step = predData_TM_five_step[nTrain:nTrain+nTest]
  # predData_shift = predData_shift[nTrain:nTrain+nTest]
  # predData_ARIMA_five_step = predData_ARIMA_five_step[nTrain:nTrain+nTest]
  # predData_LSTM_five_step = predData_LSTM_five_step[nTrain:nTrain+nTest]

  NRMSE_TM = NRMSE(trueData[nTrain:nTrain+nTest], predData_TM_five_step[nTrain:nTrain+nTest])
  NRMSE_ARIMA = NRMSE(trueData[nTrain:nTrain+nTest], predData_ARIMA_five_step[nTrain:nTrain+nTest])
  NRMSE_LSTM = NRMSE(trueData[nTrain:nTrain+nTest], predData_LSTM_five_step[nTrain:nTrain+nTest])
  NRMSE_LSTM2 = NRMSE(trueData[nTrain:nTrain+nTest], predData_LSTM_five_step2[nTrain:nTrain+nTest])
  NRMSE_Shift = NRMSE(trueData[nTrain:nTrain+nTest], predData_shift[nTrain:nTrain+nTest])


  altMAPE_TM = altMAPE(trueData[nTrain:nTrain+nTest], predData_TM_five_step[nTrain:nTrain+nTest])
  altMAPE_ARIMA = altMAPE(trueData[nTrain:nTrain+nTest], predData_ARIMA_five_step[nTrain:nTrain+nTest])
  altMAPE_LSTM = altMAPE(trueData[nTrain:nTrain+nTest], predData_LSTM_five_step[nTrain:nTrain+nTest])
  altMAPE_LSTM2 = altMAPE(trueData[nTrain:nTrain+nTest], predData_LSTM_five_step2[nTrain:nTrain+nTest])
  altMAPE_Shift = altMAPE(trueData[nTrain:nTrain+nTest], predData_shift[nTrain:nTrain+nTest])

  print "NRMSE: Shift - 5 step", NRMSE_Shift
  print "NRMSE: TM - 5 step", NRMSE_TM
  print "NRMSE: ARIMA - 5 step", NRMSE_ARIMA
  print "NRMSE: LSTM - 5 step", NRMSE_LSTM
  print "NRMSE: LSTM - 5 step, no day of week", NRMSE_LSTM2

  time_step = pd.to_datetime(time_step)

  plt.figure(2)
  plt.plot(time_step, trueData, label='True Data', color='black')
  # plt.plot(time_step, predData_shift, label='Trival NRMSE: '+"%0.3f" % NRMSE_Shift)
  # plt.plot(time_step, predData_ARIMA_five_step, label='ARIMA NRMSE: '+"%0.3f" % NRMSE_ARIMA)
  plt.plot(time_step, predData_TM_five_step, label='TM, NRMSE: '+"%0.3f" % NRMSE_TM)
  plt.plot(time_step, predData_LSTM_five_step, label='LSTM, NRMSE: '+"%0.3f" % NRMSE_LSTM)
  plt.plot(time_step, predData_LSTM_five_step2, label='LSTM no TimeOfDay, NRMSE: '+"%0.3f" % NRMSE_LSTM)
  plt.legend()
  plt.xlabel('Time')
  plt.ylabel('Passenger Count')
  plt.xlim([time_step.values[-500], time_step.values[-1]])
  fileName = './result/'+dataSet+"modelPrediction.pdf "
  # print "save example prediction trace to ", fileName
  # plt.savefig(fileName)

  fig, ax = plt.subplots(nrows=1, ncols=2)
  inds = np.arange(5)
  ax[0].bar(inds, [altMAPE_Shift, altMAPE_ARIMA, altMAPE_TM, altMAPE_LSTM, altMAPE_LSTM2], width=0.3)
  ax[0].set_xticks(inds+0.3/2)
  ax[0].set_xticklabels( ('Shift', 'ARIMA', 'TM', 'LSTM', 'LSTM-NoDayOfWeek') )
  ax[0].set_ylabel('altMAPE')

  ax[1].bar(inds, [NRMSE_Shift, NRMSE_ARIMA, NRMSE_TM, NRMSE_LSTM, NRMSE_LSTM2], width=0.3)
  ax[1].set_xticks(inds+0.3/2)
  ax[1].set_xticklabels( ('Shift', 'ARIMA', 'TM', 'LSTM', 'LSTM-NoDayOfWeek') )
  ax[1].set_ylabel('NRMSE')
  # import plotly.plotly as py
  # plot_url = py.plot_mpl(fig)




  # plot NRMSE as a function of time
  nrmse_window = 960
  (window_center, nrmse_slide_tm) = NRMSE_sliding(trueData, predData_TM_five_step, nrmse_window)
  (window_center, nrmse_slide_lstm) = NRMSE_sliding(trueData, predData_LSTM_five_step, nrmse_window)
  (window_center, nrmse_slide_lstm2) = NRMSE_sliding(trueData, predData_LSTM_five_step2, nrmse_window)
  (window_center, nrmse_slide_shift) = NRMSE_sliding(trueData, predData_shift, nrmse_window)
  (window_center, nrmse_arima_arima) = NRMSE_sliding(trueData, predData_ARIMA_five_step, nrmse_window)

  plt.figure(1)
  plt.plot(time_step[window_center], nrmse_slide_tm, label='TM')
  plt.plot(time_step[window_center], nrmse_slide_lstm, label='LSTM')
  plt.plot(time_step[window_center], nrmse_slide_lstm2, label='LSTM-NoDayOfWeek')
  plt.plot(time_step[window_center], nrmse_slide_shift, label='Shift')
  plt.plot(time_step[window_center], nrmse_arima_arima, label='ARIMA')
  plt.legend()
  plt.xlim([time_step[nTrain+nrmse_window], time_step.values[-1]])
  plt.ylim([0.2, 1.1])
  plt.xlabel(' time step ')
  plt.ylabel(' NRMSE ')


