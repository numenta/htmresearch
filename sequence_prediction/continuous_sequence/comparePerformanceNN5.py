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
import csv, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from optparse import OptionParser
rcParams.update({'figure.autolayout': True})
plt.ion()
plt.close('all')

def loadDataFile(filePath):
  reader = csv.reader(filePath)
  ncol=len(next(reader)) # Read first line and count columns
  if ncol == 1:
    data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['value'])
  else:
    data = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['step', 'value'])
  data = data['value'].astype('float').tolist()
  data = np.array(data)
  return data

def NRMSE(data, pred):
  return np.sqrt(np.nanmean(np.square(pred-data)))/np.sqrt(np.nanmean( np.square(data-np.nanmean(data))))


def plotPerformance(dataSet):
  filePath = './data/NN5/NN5-' + str(dataSet) + '.csv'
  print "load test data from ", filePath
  trueData = loadDataFile(filePath)

  filePath = './prediction/NN5-' + str(dataSet) + '_TM_pred.csv'
  print "load TM prediction from ", filePath
  predData_TM = loadDataFile(filePath)

  # filePath = './prediction/' + dataSet + '_ARIMA_pred_cont.csv'
  # predData_ARIMA = loadDataFile(filePath)
  N = min(len(predData_TM), len(trueData))

  TM_lag = 1

  predData_shift = np.roll(trueData, 1)
  predData_TM = np.roll(predData_TM, TM_lag)

  trueData = trueData[TM_lag:]
  predData_TM = predData_TM[TM_lag:]
  predData_shift = predData_shift[TM_lag:]

  # predData_ARIMA = predData_ARIMA[lag:N]

  NRMSE_TM = NRMSE(trueData, predData_TM)
  # NRMSE_ARIMA = NRMSE(trueData, predData_ARIMA)
  NRMSE_Shift = NRMSE(trueData, predData_shift)

  resTM = abs(trueData-predData_TM)
  res_shift = abs(trueData-predData_shift)
  resTM = resTM[np.isnan(resTM) == False]
  res_shift = res_shift[np.isnan(res_shift) == False]

  print "NRMSE: Shift", NRMSE_Shift
  print "NRMSE: TM", NRMSE_TM
  # print "NRMSE: ARIMA", NRMSE_ARIMA


  plt.figure(1)
  plt.plot(trueData, label='True Data')
  plt.plot(predData_shift, label='Trival NRMSE: '+"%0.3f" % NRMSE_Shift)
  # plt.plot(predData_ARIMA, label='ARIMA NRMSE: '+"%0.3f" % NRMSE_ARIMA)
  plt.plot(predData_TM, label='TM, NRMSE: '+"%0.3f" % NRMSE_TM)
  plt.legend()
  plt.xlabel('Time')
  fileName = './result/NN5'+str(dataSet)+"_modelPrediction.pdf"
  print "save example prediction trace to ", fileName
  plt.savefig(fileName)

  plt.figure(2)
  xl = [0, max(max(resTM), max(res_shift))]
  plt.subplot(2,2,1)
  plt.hist(resTM)
  plt.title('TM median='+"%0.3f" % np.median(resTM)+' NRMSE: '+"%0.3f" % NRMSE_TM)
  plt.xlim(xl)
  plt.xlabel("|residual|")
  plt.subplot(2,2,3)
  plt.hist(res_shift)
  plt.title('Trivial median='+"%0.3f" % np.median(res_shift)+' NRMSE: '+"%0.3f" % NRMSE_Shift)
  plt.xlim(xl)
  plt.xlabel("|residual|")
  fileName = './result/NN5-'+str(dataSet)+"_error_distribution.pdf"
  print "save residual error distribution to ", fileName
  plt.savefig(fileName)

def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nCompare TM performance with trivial predictor using "
                              "model outputs in prediction directory "
                              "and outputting results to result directory.")
  parser.add_option("-d",
                    "--dataSet",
                    type=int,
                    default=0,
                    dest="dataSet",
                    help="DataSet Name, choose from sine, SantaFe_A, MackeyGlass")

  (options, args) = parser.parse_args(sys.argv[1:])
  print args
  return options, args


if __name__ == "__main__":

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  print 'Compare Model performance for ', dataSet
  plotPerformance(dataSet)
