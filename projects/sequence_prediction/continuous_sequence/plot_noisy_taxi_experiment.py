import sys
import subprocess

import numpy as np
import pandas as pd

from plot import computeSquareDeviation

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')


def computeAltMAPE(truth, prediction, startFrom=0):
  return np.nanmean(np.abs(truth[startFrom:] - prediction[startFrom:]))/np.nanmean(np.abs(truth[startFrom:]))


def computeNRMSE(truth, prediction, startFrom=0):
  square_deviation = computeSquareDeviation(prediction, truth)
  square_deviation[:startFrom] = None
  return np.sqrt(np.nanmean(square_deviation))/np.nanstd(truth)


# use datetime as x-axis
dataSet = 'nyc_taxi'
filePath = './data/' + dataSet + '.csv'
data = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['datetime', 'value', 'timeofday', 'dayofweek'])

xaxis_datetime = pd.to_datetime(data['datetime'])

startFrom = 6000
window = 960


noiseList = [0, 0.01, 0.02, 0.04, 0.06, 0.08]

nrmse = pd.DataFrame([], columns=['TM'])
mape = pd.DataFrame([], columns=['TM'])
for noise in noiseList:
  if noise > 0:
    filePath = './prediction/nyc_taxi' + "noise_{:.2f}".format(noise) + '_TM_pred.csv'
  else:
    filePath = './prediction/nyc_taxi_TM_pred.csv'

  predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
  tm_truth = np.roll(predData_TM['value'], -5)
  predData_TM_five_step = np.array(predData_TM['prediction5'])

  nrmseTM = computeNRMSE(tm_truth, predData_TM_five_step, startFrom)
  altMAPETM = computeAltMAPE(tm_truth, predData_TM_five_step, startFrom)

  nrmse = pd.concat([nrmse,
                     pd.DataFrame([nrmseTM], columns=['TM'])])

  mape = pd.concat([mape,
                     pd.DataFrame([altMAPETM], columns=['TM'])])

plt.figure()
plt.plot(noiseList, mape)
plt.xlabel(' Noise Amount ')
plt.ylabel(' MAPE')

plt.figure()
plt.plot(noiseList, nrmse)
plt.xlabel(' Noise Amount ')
plt.ylabel(' NRMSE')