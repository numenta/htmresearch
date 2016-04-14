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
import numpy as np
from pylab import rcParams
from plot import ExperimentResult, plotAccuracy, computeSquareDeviation, computeLikelihood, plotLSTMresult
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

xaxis_datetime = pd.to_datetime(data['datetime'])

expResult_perturb = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window'+str(1001.0)+'/')
negLL_LSTM1000_perturb = expResult_perturb.error
truth_LSTM1000_perturb = expResult_perturb.truth

expResult_perturb = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window'+str(3001.0)+'/')
negLL_LSTM3000_perturb = expResult_perturb.error
truth_LSTM3000_perturb = expResult_perturb.truth

expResult_perturb = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window'+str(6001.0)+'/')
negLL_LSTM6000_perturb = expResult_perturb.error
truth_LSTM6000_perturb = expResult_perturb.truth


dataSet = 'nyc_taxi_perturb'
tm_prediction_perturb = np.load('./result/'+dataSet+'TMprediction.npy')
tm_truth_perturb = np.load('./result/'+dataSet+'TMtruth.npy')
from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)

filePath = './prediction/' + dataSet + '_TM_pred.csv'
predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])

predData_TM_five_step = np.array(predData_TM['prediction5'])
iteration = predData_TM.index

tm_pred_perturb_truth = np.roll(predData_TM['value'], -5)
tm_pred_perturb = np.array(predData_TM['prediction5'])


filePath = './prediction/' + dataSet + '_esn_pred.csv'
predDataESN = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                          names=['step', 'value', 'prediction5'])
esnPredPerturbTruth = np.roll(predDataESN['value'], -5)
esnPredPerturb = np.array(predDataESN['prediction5'])


negLL_tm_perturb = computeLikelihood(tm_prediction_perturb, tm_truth_perturb, encoder)
negLL_tm_perturb[:6000] = None
nrmse_tm_perturb = computeSquareDeviation(tm_pred_perturb, tm_pred_perturb_truth)
mape_tm_perturb = np.abs(tm_pred_perturb - tm_pred_perturb_truth)
mape_esn_perturb = np.abs(esnPredPerturb - esnPredPerturbTruth)

plt.figure()
plotAccuracy((negLL_LSTM3000_perturb, xaxis_datetime), truth_LSTM3000_perturb,
             window=window, errorType='negLL', label='LSTM3000', train=expResult_perturb.train)
# plotAccuracy((negLL_LSTM3000_perturb_baseline, xaxis_datetime), truth_LSTM3000_perturb, window=window, errorType='negLL', label='TM')

plotAccuracy((negLL_LSTM6000_perturb, xaxis_datetime), truth_LSTM6000_perturb, window=window, errorType='negLL', label='LSTM6000')
# plotAccuracy((negLL_LSTM6000_perturb_baseline, xaxis_datetime), truth_LSTM3000_perturb, window=window, errorType='negLL', label='TM')

plotAccuracy((negLL_tm_perturb, xaxis_datetime), tm_truth_perturb, window=window, errorType='negLL', label='TM')
plt.axvline(xaxis_datetime[13152], color='black', linestyle='--')
plt.xlim([xaxis_datetime[13000], xaxis_datetime[15000]])
plt.legend()
plt.ylim([1.2, 2.3])
plt.ylabel('Negative Log-Likelihood')
plt.savefig(figPath + 'example_perturbation.pdf')


expResult_perturb_1000 = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_perturb/learning_window'+str(1001.0)+'/')

expResult_perturb_3000 = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_perturb/learning_window'+str(3001.0)+'/')

expResult_perturb_6000 = ExperimentResult(
  'results/nyc_taxi_experiment_continuous_perturb/learning_window'+str(6001.0)+'/')

nrmse_LSTM1000_perturb = expResult_perturb_1000.error
nrmse_LSTM3000_perturb = expResult_perturb_3000.error
nrmse_LSTM6000_perturb = expResult_perturb_6000.error
mape_LSTM1000_perturb = np.abs(expResult_perturb_1000.truth - expResult_perturb_1000.predictions)
mape_LSTM3000_perturb = np.abs(expResult_perturb_3000.truth - expResult_perturb_3000.predictions)
mape_LSTM6000_perturb = np.abs(expResult_perturb_6000.truth - expResult_perturb_6000.predictions)


plt.figure()
window = 400
plotAccuracy((mape_LSTM1000_perturb, xaxis_datetime), truth_LSTM3000_perturb,
             window=window, errorType='mape', label='LSTM1000', train=expResult_perturb.train)

plotAccuracy((mape_LSTM3000_perturb, xaxis_datetime), truth_LSTM3000_perturb,
             window=window, errorType='mape', label='LSTM3000', train=expResult_perturb.train)

plotAccuracy((mape_LSTM6000_perturb, xaxis_datetime), truth_LSTM6000_perturb,
             window=window, errorType='mape', label='LSTM6000')
plotAccuracy((mape_esn_perturb, xaxis_datetime), esnPredPerturbTruth,
             window=window, errorType='mape', label='ESN')

plotAccuracy((mape_tm_perturb, xaxis_datetime), tm_truth_perturb,
             window=window, errorType='mape', label='TM')

plt.axvline(xaxis_datetime[13152], color='black', linestyle='--')
plt.xlim([xaxis_datetime[13000], xaxis_datetime[15000]])
plt.legend()
plt.ylim([.1, .4])
plt.ylabel('MAPE')
plt.savefig(figPath + 'example_perturbation_MAPE.pdf')

startFrom = 13152
endAt = startFrom+1440
norm_factor = np.nanstd(tm_truth_perturb[startFrom:endAt])

fig, ax = plt.subplots(nrows=1, ncols=3)
inds = np.arange(4)

width = 0.5

ax1 = ax[0]
ax1.bar(inds, [np.sqrt(np.nanmean(nrmse_LSTM1000_perturb[startFrom:endAt]))/norm_factor,
               np.sqrt(np.nanmean(nrmse_LSTM3000_perturb[startFrom:endAt]))/norm_factor,
               np.sqrt(np.nanmean(nrmse_LSTM6000_perturb[startFrom:endAt]))/norm_factor,
               np.sqrt(np.nanmean(nrmse_tm_perturb[startFrom:endAt]))/norm_factor], width=width)
ax1.set_xticks(inds+width/2)
ax1.set_xticklabels( ('LSTM1000', 'LSTM3000', 'LSTM6000', 'TM') )
ax1.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
ax1.set_ylabel('NRMSE')

ax2 = ax[1]
width = 0.5
norm_factor = np.nanmean(np.abs(tm_truth_perturb[startFrom:endAt]))
ax2.bar(inds, [np.nanmean(mape_LSTM1000_perturb[startFrom:endAt])/norm_factor,
               np.nanmean(mape_LSTM3000_perturb[startFrom:endAt])/norm_factor,
               np.nanmean(mape_LSTM6000_perturb[startFrom:endAt]/norm_factor),
               np.nanmean(mape_tm_perturb[startFrom:endAt])/norm_factor], width=width)
ax2.set_xticks(inds+width/2)
ax2.set_xticklabels( ('LSTM1000', 'LSTM3000', 'LSTM6000', 'TM') )
ax2.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
ax2.set_ylabel('MAPE')

ax3 = ax[2]
width = 0.5
ax3.bar(inds, [np.nanmean(negLL_LSTM1000_perturb[startFrom:endAt]),
               np.nanmean(negLL_LSTM3000_perturb[startFrom:endAt]),
               np.nanmean(negLL_LSTM6000_perturb[startFrom:endAt]),
               np.nanmean(negLL_tm_perturb[startFrom:])], width=width)
ax3.set_xticks(inds+width/2)
ax3.set_xticklabels( ('LSTM1000', 'LSTM3000', 'LSTM6000', 'TM') )
ax3.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
ax3.set_ylabel('Negative Log-likelihood')
plt.savefig(figPath + 'model_performance_after_perturbation.pdf')


#
# def getPerturbBaseline(window, errorType='likelihood'):
#   if errorType == 'likelihood':
#     expResult_unperturb = ExperimentResult(
#       'results/nyc_taxi_experiment_continuous_likelihood/learning_window'+str(window)+'/')
#     expResult_perturb = ExperimentResult(
#       'results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window'+str(window)+'/')
#     expResult_perturb_baseline = ExperimentResult(
#       'results/nyc_taxi_experiment_continuous_likelihood_perturb_baseline/learning_window'+str(window)+'/')
#   elif errorType == 'NRMSE' or errorType == 'MAPE':
#     expResult_unperturb = ExperimentResult(
#       'results/nyc_taxi_experiment_continuous/learning_window'+str(window)+'/')
#     expResult_perturb = ExperimentResult(
#       'results/nyc_taxi_experiment_continuous_perturb/learning_window'+str(window)+'/')
#     expResult_perturb_baseline = ExperimentResult(
#       'results/nyc_taxi_experiment_continuous_perturb_baseline/learning_window'+str(window)+'/')
#
#   truth = expResult_unperturb.truth
#   error_perturb = expResult_perturb.error
#   error_perturb_baseline = expResult_perturb_baseline.error
#   error_perturb_baseline[:13152] = expResult_unperturb.error[:13152]
#
#   if errorType == 'MAPE':
#     error_perturb = np.abs(expResult_perturb.predictions - expResult_perturb.truth)
#     error_unperturb = np.abs(expResult_unperturb.predictions - expResult_unperturb.truth)
#     error_perturb_baseline= np.abs(expResult_perturb_baseline.predictions - expResult_perturb_baseline.truth)
#     error_perturb_baseline[:13152] = error_unperturb[:13152]
#
#   return (error_perturb, error_perturb_baseline, truth)
#
#

#
# ### Figure 2: Continuous LSTM with different window size
#
# fig = plt.figure()
# (NRMSE_LSTM1000, expResult_LSTM1000) = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window1001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-1000')
#
# (NRMSE_LSTM3000, expResult_LSTM3000) = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window3001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-3000')
#
# (NRMSE_LSTM6000, expResult_LSTM6000) = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous/learning_window6001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-6000')
#
# dataSet = 'nyc_taxi'
# filePath = './prediction/' + dataSet + '_TM_pred.csv'
# predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
# tm_truth = np.roll(predData_TM['value'], -5)
# predData_TM_five_step = np.array(predData_TM['prediction5'])
#
# square_deviation = computeSquareDeviation(predData_TM_five_step, tm_truth)
# square_deviation[:6000] = None
#
# NRMSE_TM = plotAccuracy((square_deviation, xaxis_datetime),
#                        tm_truth,
#                        window=window,
#                        errorType='square_deviation',
#                        label='TM')
#
# ### Figure: Continuous LSTM & TM on perturbed data
# fig = plt.figure()
# plt.clf()
# (NRMSE_LSTM1000_perturb, expResult_LSTM1000_perturb) = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous_perturb/learning_window1001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-1000')
#
# (NRMSE_LSTM3000_perturb, expResult_LSTM3000_perturb) = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous_perturb/learning_window3001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-3000')
#
# (NRMSE_LSTM6000_perturb, expResult_LSTM6000_perturb) = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous_perturb/learning_window6001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-6000')
#
# # load TM prediction
# filePath = './data/' + 'nyc_taxi' + '.csv'
# data = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['datetime', 'value', 'timeofday', 'dayofweek'])
#
# dataSet = 'nyc_taxi_perturb'
# filePath = './prediction/' + dataSet + '_TM_pred.csv'
# predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
# tm_truth = np.roll(predData_TM['value'], -5)
# predData_TM_five_step = np.array(predData_TM['prediction5'])
# iteration = predData_TM.index
#
# square_deviation_perturb = computeSquareDeviation(predData_TM_five_step, tm_truth)
# square_deviation_perturb[:6000] = None
# NRMSE_TM_perturb = plotAccuracy((square_deviation_perturb, xaxis_datetime),
#                        tm_truth,
#                        window=window,
#                        errorType='square_deviation',
#                        label='TM')
# plt.legend()
# plt.savefig(figPath + 'continuous_perturb.pdf')
#
#
# ###
# fig = plt.figure()
# # window = 960
# plt.clf()
# NRMSE_diff_TM = NRMSE_TM_perturb - NRMSE_TM
# NRMSE_diff_TM = NRMSE_diff_TM/np.nanstd(NRMSE_diff_TM)
#
# NRMSE_diff_LSTM6000 = NRMSE_LSTM6000_perturb - NRMSE_LSTM6000
# NRMSE_diff_LSTM6000 = NRMSE_diff_LSTM6000/np.nanstd(NRMSE_diff_LSTM6000)
#
# NRMSE_diff_LSTM3000 = NRMSE_LSTM3000_perturb - NRMSE_LSTM3000
# NRMSE_diff_LSTM3000 = NRMSE_diff_LSTM3000/np.nanstd(NRMSE_diff_LSTM3000)
#
# NRMSE_diff_LSTM1000 = NRMSE_LSTM1000_perturb - NRMSE_LSTM1000
# NRMSE_diff_LSTM1000 = NRMSE_diff_LSTM1000/np.nanstd(NRMSE_diff_LSTM1000)
#
# plotAccuracy((NRMSE_diff_TM, xaxis_datetime),
#              tm_truth, window=window, errorType='negLL', label='TM')
# plotAccuracy((NRMSE_diff_LSTM6000, xaxis_datetime),
#              tm_truth, window=window, errorType='negLL', label='LSTM-6000')
# plotAccuracy((NRMSE_diff_LSTM3000, xaxis_datetime),
#              tm_truth, window=window, errorType='negLL', label='LSTM-3000')
# plotAccuracy((NRMSE_diff_LSTM1000, xaxis_datetime),
#              tm_truth, window=window, errorType='negLL', label='LSTM-1000')
# plt.ylabel(' difference of squared deviation')
# plt.legend()
#
#
#
# ### Figure 4: Continuous LSTM with different window size using the likelihood metric
#
# fig = plt.figure()
# # negLL_StaticLSTM = \
# #   plotLSTMresult('results/nyc_taxi_experiment_one_shot_likelihood/',
# #                window, xaxis=xaxis_datetime, label='static LSTM ')
#
# plt.clf()
# (negLL_LSTM1000, expResult_LSTM1000_negLL) = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window1001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-1000')
#
# (negLL_LSTM3000, expResult_LSTM3000_negLL) = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window3001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-3000')
#
# (negLL_LSTM6000, expResult_LSTM6000_negLL) = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window6001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-6000')
#
# dataSet = 'nyc_taxi'
# tm_prediction = np.load('./result/'+dataSet+'TMprediction.npy')
# tm_truth = np.load('./result/'+dataSet+'TMtruth.npy')
# from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
# encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
# negLL = computeLikelihood(tm_prediction, tm_truth, encoder)
# negLL[:6000] = None
# negLL_TM = \
#   plotAccuracy((negLL, xaxis_datetime), tm_truth, window=window, errorType='negLL', label='TM')
# plt.legend()
# plt.savefig(figPath + 'continuous_likelihood.pdf')
#
#
# ### Figure 4: Perturbed data
#
# fig = plt.figure(5)
# plt.clf()
# negLL_LSTM1000_perturb = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window1001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-1000')
#
# negLL_LSTM3000_perturb = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window3001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-3000')
#
# negLL_LSTM6000_perturb = \
#   plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window6001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-6000')
#
# dataSet = 'nyc_taxi_perturb'
# tm_prediction_perturb = np.load('./result/'+dataSet+'TMprediction.npy')
# tm_truth_perturb = np.load('./result/'+dataSet+'TMtruth.npy')
# from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
# encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
# negLL_tm_perturb = computeLikelihood(tm_prediction_perturb, tm_truth_perturb, encoder)
# negLL_tm_perturb[:6000] = None
# plotAccuracy((negLL_tm_perturb, xaxis_datetime), tm_truth_perturb, window=window, errorType='negLL', label='TM')
# plt.legend()
#
#
# dataSet = 'nyc_taxi_perturb_baseline'
# tm_prediction_perturb = np.load('./result/'+dataSet+'TMprediction.npy')
# tm_truth_perturb = np.load('./result/'+dataSet+'TMtruth.npy')
# from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
# encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
# negLL_tm_perturb_baseline = computeLikelihood(tm_prediction_perturb, tm_truth_perturb, encoder)
# negLL_tm_perturb_baseline[:6000] = None
#
#
# plt.figure()
# plotAccuracy((negLL_tm_perturb, xaxis_datetime), tm_truth_perturb, window=window, errorType='negLL', label='TM')
# negLL_tm_perturb_baseline[:13152] = negLL_TM[:13152]
# plotAccuracy((negLL_tm_perturb_baseline, xaxis_datetime), tm_truth_perturb, window=window, errorType='negLL', label='TM')
# plt.figure()
# plotAccuracy((negLL_tm_perturb-negLL_tm_perturb_baseline, xaxis_datetime), tm_truth_perturb, window=window, errorType='negLL', label='TM')
#
# plt.savefig(figPath + 'continuous_likelihood_perturb.pdf')
#
# ###
# fig = plt.figure(8)
# plt.clf()
# plotAccuracy((negLL_tm_perturb - negLL, xaxis_datetime), tm_truth, window=window, errorType='negLL', label='TM')
# # plotAccuracy((negLL_LSTM6000_perturb - negLL_LSTM6000, xaxis_datetime),
# #              tm_truth, window=window, errorType='negLL', label='LSTM-6000')
# plotAccuracy((negLL_LSTM3000_perturb - negLL_LSTM3000, xaxis_datetime),
#              tm_truth, window=window, errorType='negLL', label='LSTM-3000')
# plotAccuracy((negLL_LSTM1000_perturb - negLL_LSTM1000, xaxis_datetime),
#              tm_truth, window=window, errorType='negLL', label='LSTM-1000')
# plt.legend()
#
# plt.figure()
# plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood_perturb/learning_window3001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-3000')
#
# plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood_perturb_baseline/learning_window3001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-3000')
#
# plotLSTMresult('results/nyc_taxi_experiment_continuous_likelihood/learning_window3001.0/',
#                window, xaxis=xaxis_datetime, label='continuous LSTM-3000')
#
# (negLL_LSTM3000_perturb, negLL_LSTM3000_perturb_baseline, truth_LSTM3000_perturb) = getPerturbBaseline(3001.0)
# (negLL_LSTM6000_perturb, negLL_LSTM6000_perturb_baseline, truth_LSTM6000_perturb) = getPerturbBaseline(6001.0)
#
#
# (nrmse_LSTM3000_perturb, nrmse_LSTM3000_perturb_baseline, truth_LSTM3000_perturb) = getPerturbBaseline(3001.0, errorType='NRMSE')
# (nrmse_LSTM6000_perturb, nrmse_LSTM6000_perturb_baseline, truth_LSTM6000_perturb) = getPerturbBaseline(6001.0, errorType='NRMSE')
#
#
# (mape_LSTM3000_perturb, mape_LSTM3000_perturb_baseline, truth_LSTM3000_perturb) = getPerturbBaseline(3001.0, errorType='MAPE')
# (mape_LSTM6000_perturb, mape_LSTM6000_perturb_baseline, truth_LSTM6000_perturb) = getPerturbBaseline(6001.0, errorType='MAPE')
#
#
# filePath = './prediction/' + 'nyc_taxi' + '_TM_pred.csv'
# predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
# tm_truth = np.roll(predData_TM['value'], -5)
# tm_pred_unperturb = np.array(predData_TM['prediction5'])
#
# filePath = './prediction/' + 'nyc_taxi_perturb' + '_TM_pred.csv'
# predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
# tm_pred_perturb_truth = np.roll(predData_TM['value'], -5)
# tm_pred_perturb = np.array(predData_TM['prediction5'])
#
# filePath = './prediction/' + 'nyc_taxi_perturb_baseline' + '_TM_pred.csv'
# predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['step', 'value', 'prediction5'])
# tm_pred_perturb_baseline = np.array(predData_TM['prediction5'])
#
# nrmse_unperturb = computeSquareDeviation(tm_pred_unperturb, tm_truth)
# nrmse_tm_perturb = computeSquareDeviation(tm_pred_perturb, tm_truth)
# nrmse_tm_perturb_baseline = computeSquareDeviation(tm_pred_perturb_baseline, tm_truth)
# nrmse_tm_perturb_baseline[:13152] = nrmse_unperturb[:13152]
#
# mape_tm_perturb = np.abs(tm_pred_perturb - tm_pred_perturb_truth)
#
# plt.figure()
# window = 480
# plotAccuracy((nrmse_LSTM3000_perturb, xaxis_datetime), truth_LSTM3000_perturb, window=window, errorType='square_deviation', label='LSTM3000')
# # plotAccuracy((nrmse_LSTM3000_perturb_baseline, xaxis_datetime), truth_LSTM3000_perturb, window=window, errorType='nrmse', label='TM')
#
# plotAccuracy((nrmse_LSTM6000_perturb, xaxis_datetime), truth_LSTM6000_perturb, window=window, errorType='square_deviation', label='LSTM6000')
# # plotAccuracy((nrmse_LSTM6000_perturb_baseline, xaxis_datetime), truth_LSTM3000_perturb, window=window, errorType='nrmse', label='TM')
#
# plotAccuracy((nrmse_tm_perturb, xaxis_datetime), tm_truth_perturb, window=window, errorType='square_deviation', label='TM')
# plt.axvline(xaxis_datetime[13152], color='black', linestyle='--')
# plt.xlim([xaxis_datetime[12000], xaxis_datetime[15000]])
# plt.legend()
#



# plt.figure()
# window = 960
# plotAccuracy((negLL_LSTM3000_perturb-negLL_LSTM3000_perturb_baseline, xaxis_datetime),
#              truth_LSTM3000_perturb, window=window, errorType='negLL', label='LSTM-3000')
# plotAccuracy((negLL_LSTM6000_perturb-negLL_LSTM6000_perturb_baseline, xaxis_datetime),
#              truth_LSTM6000_perturb, window=window, errorType='negLL', label='LSTM-6000')
# plotAccuracy((negLL_tm_perturb-negLL_tm_perturb_baseline, xaxis_datetime),
#              tm_truth_perturb, window=window, errorType='negLL', label='TM')
# plt.axvline(xaxis_datetime[13152], color='black', linestyle='--')
# plt.xlim([xaxis_datetime[12000], xaxis_datetime[15000]])
# # plt.xlim([xaxis_datetime[12800], xaxis_datetime[17519]])
# plt.legend()
