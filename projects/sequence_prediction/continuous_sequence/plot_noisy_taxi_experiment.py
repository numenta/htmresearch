import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from plot import computeSquareDeviation
from plot import ExperimentResult
from plot import computeLikelihood
from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder

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
data = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                   names=['datetime', 'value', 'timeofday', 'dayofweek'])

xaxis_datetime = pd.to_datetime(data['datetime'])

startFrom = 10000
noiseList = [0, 0.02, 0.04, 0.06, 0.08, 0.1]


encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)


nrmseTM = pd.DataFrame([], columns=['TM'])
mapeTM = pd.DataFrame([], columns=['TM', 'GT'])
negLLTM = pd.DataFrame([], columns=['TM'])
noiseStrengthTM = []
for noise in noiseList:
  if noise > 0:
    dataSet = 'nyc_taxi' + "noise_{:.2f}".format(noise)
    filePath = './prediction/nyc_taxi' + "noise_{:.2f}".format(noise) + '_TM_pred.csv'
  else:
    dataSet = 'nyc_taxi'
    filePath = './prediction/nyc_taxi_TM_pred.csv'

  predData_TM = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                            names=['step', 'value', 'prediction5'])

  noiseValue = (predData_TM['value'] - data['value']) / data['value']

  noiseStrengthTM.append(np.std(noiseValue[8000:]))

  if not np.isclose(noiseStrengthTM[-1], float(noise), rtol=0.1):
    print "Warning: Estimated noise strength is different from the given noise"


  groundTruth = np.roll(data['value'], -5)
  tmTruth = np.roll(predData_TM['value'], -5)
  predDataTMFiveStep = np.array(predData_TM['prediction5'])

  nrmse = computeNRMSE(tmTruth, predDataTMFiveStep, startFrom)
  altMAPE = computeAltMAPE(tmTruth, predDataTMFiveStep, startFrom)

  mapeGroundTruth = computeAltMAPE(tmTruth, groundTruth, startFrom)

  tm_prediction = np.load('./result/'+dataSet+'TMprediction.npy')
  tmTruth = np.load('./result/' + dataSet + 'TMtruth.npy')
  negLL = computeLikelihood(tm_prediction, tmTruth, encoder)
  negLL = np.nanmean(negLL[startFrom:])

  nrmseTM = pd.concat([nrmseTM,
                       pd.DataFrame([nrmse], columns=['TM'])])

  mapeTM = pd.concat([mapeTM,
                      pd.DataFrame(np.reshape(np.array([altMAPE, mapeGroundTruth]), newshape=(1,2)),
                                   columns=['TM', 'GT'])])

  negLLTM = pd.concat([negLLTM,
                      pd.DataFrame([negLL], columns=['TM'])])



lstmExptDir = 'results/nyc_taxi_experiment_continuous_likelihood_noise/'
noiseList = ['0.0', '0.020', '0.040', '0.060', '0.080', '0.10']

negLLLSTM = pd.DataFrame([], columns=['LSTM'])
noiseStrengthLSTM = []
for noise in noiseList:
  experiment = lstmExptDir + 'noise' + noise
  expResult = ExperimentResult(experiment)
  truth = np.concatenate((np.zeros(5333), expResult.truth))
  error = np.concatenate((np.zeros(5333), expResult.error))

  noiseValue = (truth - data['value']) / data['value']
  noiseStrengthLSTM.append(np.std(noiseValue[8000:]))
  if not np.isclose(noiseStrengthLSTM[-1], float(noise), rtol=0.1):
    print "Warning: Estimated noise strength is different from the given noise"

  negLL = np.nanmean(error[startFrom:])

  negLLLSTM = pd.concat([negLLLSTM,
                         pd.DataFrame([negLL], columns=['LSTM'])])



lstmExptDir = 'results/nyc_taxi_experiment_continuous_noise/'
noiseList = ['0.0', '0.020', '0.040', '0.060', '0.080', '0.10']

mapeLSTM = pd.DataFrame([], columns=['LSTM'])
nrmseLSTM = pd.DataFrame([], columns=['LSTM'])
for noise in noiseList:
  experiment = lstmExptDir + 'noise' + noise
  expResult = ExperimentResult(experiment)

  altMAPE = computeAltMAPE(expResult.truth,
                           expResult.predictions, startFrom)

  nrmse = computeNRMSE(expResult.truth,
                           expResult.predictions, startFrom)

  nrmseLSTM = pd.concat([nrmseLSTM,
                        pd.DataFrame([nrmse], columns=['LSTM'])])


  mapeLSTM = pd.concat([mapeLSTM,
                        pd.DataFrame([altMAPE], columns=['LSTM'])])


plt.figure()
plt.plot(noiseList, mapeTM['TM'])
plt.plot(noiseList, mapeLSTM)
plt.xlabel(' Noise Amount ')
plt.ylabel(' MAPE')
plt.legend(['HTM', 'LSTM'], loc=2)


plt.figure()
plt.plot(noiseList, nrmseTM)
plt.plot(noiseList, nrmseLSTM)
plt.xlabel(' Noise Amount ')
plt.legend(['HTM', 'LSTM'], loc=2)
plt.ylabel(' NRMSE')


plt.figure()
plt.plot(noiseList, negLLTM)
plt.plot(noiseList, negLLLSTM)
plt.xlabel(' Noise Amount ')
plt.legend(['HTM', 'LSTM'], loc=2)
plt.ylabel(' negLL ')


plt.figure()
plt.plot(data['value'])
plt.plot(predData_TM['value'])
plt.legend(['Original', 'Noisy'])
plt.title(" Example Noisy Data")
plt.xlim([16000, 16500])
