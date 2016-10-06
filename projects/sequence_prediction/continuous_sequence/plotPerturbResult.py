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

from htmresearch.support.sequence_learning_utils import *
import pandas as pd
import numpy as np
from pylab import rcParams
from plot import ExperimentResult, plotAccuracy, computeSquareDeviation, computeLikelihood
from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
import plotly.plotly as py

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

xaxisDatetime = pd.to_datetime(data['datetime'])

expResultPerturb = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window'+str(1001.0)+'/')
negLLLSTM1000Perturb = expResultPerturb.error
truthLSTM1000Perturb = expResultPerturb.truth

expResultPerturb = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window'+str(3001.0)+'/')
negLLLSTM3000Perturb = expResultPerturb.error
truthLSTM3000Perturb = expResultPerturb.truth

expResultPerturb = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window'+str(6001.0)+'/')
negLLLSTM6000Perturb = expResultPerturb.error
truthLSTM6000Perturb = expResultPerturb.truth

expResultPerturb = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_likelihood_perturb_online/learning_window'+str(100.0)+'/')
negLLLSTMonlinePerturb = expResultPerturb.error
truth_LSTMonline_perturb = expResultPerturb.truth


dataSet = 'nyc_taxi_perturb'
tmPredictionPerturb = np.load('./result/' + dataSet + 'TMprediction.npy')
tmTruthPerturb = np.load('./result/' + dataSet + 'TMtruth.npy')

encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)

filePath = './prediction/' + dataSet + '_TM_pred.csv'
predDataTM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])

predDataTMfiveStep = np.array(predDataTM['prediction5'])
iteration = predDataTM.index

tmPredPerturbTruth = np.roll(predDataTM['value'], -5)
tmPredPerturb = np.array(predDataTM['prediction5'])


filePath = './prediction/' + dataSet + '_esn_pred.csv'
predDataESN = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                          names=['step', 'value', 'prediction5'])
esnPredPerturbTruth = np.roll(predDataESN['value'], -5)
esnPredPerturb = np.array(predDataESN['prediction5'])


negLLTMPerturb = computeLikelihood(tmPredictionPerturb, tmTruthPerturb, encoder)
negLLTMPerturb[:6000] = None
nrmseTMPerturb = computeSquareDeviation(tmPredPerturb, tmPredPerturbTruth)
mapeTMPerturb = np.abs(tmPredPerturb - tmPredPerturbTruth)
mapeESNPerturb = np.abs(esnPredPerturb - esnPredPerturbTruth)



expResultPerturb1000 = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_perturb/learning_window'+str(1001.0)+'/')

expResultPerturb3000 = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_perturb/learning_window'+str(3001.0)+'/')

expResultPerturb6000 = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_perturb/learning_window'+str(6001.0)+'/')

expResultPerturbOnline = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_perturb_online/learning_window'+str(200.0)+'/')

nrmseLSTM1000Perturb = expResultPerturb1000.error
nrmseLSTM3000Perturb = expResultPerturb3000.error
nrmseLSTM6000Perturb = expResultPerturb6000.error
nrmseLSTMOnlinePerturb = expResultPerturbOnline.error
mapeLSTM1000Perturb = np.abs(expResultPerturb1000.truth - expResultPerturb1000.predictions)
mapeLSTM3000Perturb = np.abs(expResultPerturb3000.truth - expResultPerturb3000.predictions)
mapeLSTM6000Perturb = np.abs(expResultPerturb6000.truth - expResultPerturb6000.predictions)
mapeLSTMOnlinePerturb = np.abs(expResultPerturbOnline.truth - expResultPerturbOnline.predictions)

plt.figure()
window = 400
plotAccuracy((mapeLSTM1000Perturb, xaxisDatetime), truthLSTM3000Perturb,
             window=window, errorType='mape', label='LSTM1000', train=expResultPerturb1000.train)

plotAccuracy((mapeLSTM3000Perturb, xaxisDatetime), truthLSTM3000Perturb,
             window=window, errorType='mape', label='LSTM3000')

plotAccuracy((mapeLSTM6000Perturb, xaxisDatetime), truthLSTM6000Perturb,
             window=window, errorType='mape', label='LSTM6000')

plotAccuracy((mapeLSTMOnlinePerturb, xaxisDatetime), truth_LSTMonline_perturb,
             window=window, errorType='mape', label='LSTM-online')

plotAccuracy((mapeTMPerturb, xaxisDatetime), tmTruthPerturb,
             window=window, errorType='mape', label='TM')

plt.axvline(xaxisDatetime[13152], color='black', linestyle='--')
plt.xlim([xaxisDatetime[13000], xaxisDatetime[15000]])
plt.legend()
plt.ylim([.1, .4])
plt.ylabel('MAPE')
plt.savefig(figPath + 'example_perturbation_MAPE.pdf')



plt.figure()
plotAccuracy((negLLLSTM3000Perturb, xaxisDatetime), truthLSTM3000Perturb,
             window=window, errorType='negLL', label='LSTM3000')
# plotAccuracy((negLL_LSTM3000_perturb_baseline, xaxis_datetime), truth_LSTM3000_perturb, window=window, errorType='negLL', label='TM')

plotAccuracy((negLLLSTM6000Perturb, xaxisDatetime), truthLSTM6000Perturb, window=window, errorType='negLL', label='LSTM6000')
plotAccuracy((negLLLSTMonlinePerturb, xaxisDatetime), truthLSTM6000Perturb, window=window, errorType='negLL', label='LSTM-online')
# plotAccuracy((negLL_LSTM6000_perturb_baseline, xaxis_datetime), truth_LSTM3000_perturb, window=window, errorType='negLL', label='TM')

plotAccuracy((negLLTMPerturb, xaxisDatetime), tmTruthPerturb, window=window, errorType='negLL', label='TM')
plt.axvline(xaxisDatetime[13152], color='black', linestyle='--')
plt.xlim([xaxisDatetime[13000], xaxisDatetime[15000]])
plt.legend()
plt.ylim([1.2, 2.3])
plt.ylabel('Negative Log-Likelihood')
plt.savefig(figPath + 'example_perturbation.pdf')


startFrom = 13152
endAt = startFrom+17520
norm_factor = np.nanstd(tmTruthPerturb[startFrom:endAt])

fig, ax = plt.subplots(nrows=1, ncols=3)
inds = np.arange(5)

width = 0.5

ax1 = ax[0]
ax1.bar(inds, [np.sqrt(np.nanmean(nrmseLSTMOnlinePerturb[startFrom:endAt])) / norm_factor,
               np.sqrt(np.nanmean(nrmseLSTM1000Perturb[startFrom:endAt])) / norm_factor,
               np.sqrt(np.nanmean(nrmseLSTM3000Perturb[startFrom:endAt])) / norm_factor,
               np.sqrt(np.nanmean(nrmseLSTM6000Perturb[startFrom:endAt])) / norm_factor,
               np.sqrt(np.nanmean(nrmseTMPerturb[startFrom:endAt])) / norm_factor], width=width)
ax1.set_xticks(inds+width/2)
ax1.set_xticklabels( ('LSTMonline', 'LSTM1000', 'LSTM3000', 'LSTM6000', 'TM') )
ax1.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
ax1.set_ylabel('NRMSE')

ax2 = ax[1]
width = 0.5
norm_factor = np.nanmean(np.abs(tmTruthPerturb[startFrom:endAt]))
ax2.bar(inds, [np.nanmean(mapeLSTMOnlinePerturb[startFrom:endAt]) / norm_factor,
               np.nanmean(mapeLSTM1000Perturb[startFrom:endAt]) / norm_factor,
               np.nanmean(mapeLSTM3000Perturb[startFrom:endAt]) / norm_factor,
               np.nanmean(mapeLSTM6000Perturb[startFrom:endAt] / norm_factor),
               np.nanmean(mapeTMPerturb[startFrom:endAt]) / norm_factor], width=width)
ax2.set_xticks(inds+width/2)
ax2.set_xticklabels( ('LSTMonline', 'LSTM1000', 'LSTM3000', 'LSTM6000', 'TM') )
ax2.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
ax2.set_ylabel('MAPE')

ax3 = ax[2]
width = 0.5
ax3.bar(inds, [np.nanmean(negLLLSTMonlinePerturb[startFrom:endAt]),
               np.nanmean(negLLLSTM1000Perturb[startFrom:endAt]),
               np.nanmean(negLLLSTM3000Perturb[startFrom:endAt]),
               np.nanmean(negLLLSTM6000Perturb[startFrom:endAt]),
               np.nanmean(negLLTMPerturb[startFrom:])], width=width)
ax3.set_xticks(inds+width/2)
ax3.set_xticklabels( ('LSTMonline', 'LSTM1000', 'LSTM3000', 'LSTM6000', 'TM') )
ax3.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
ax3.set_ylabel('Negative Log-likelihood')
plt.savefig(figPath + 'model_performance_after_perturbation.pdf')