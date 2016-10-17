## ----------------------------------------------------------------------
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


import importlib

from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager
from nupic.frameworks.opf import metrics
from htmresearch.frameworks.opf.clamodel_custom import CLAModel_custom
import nupic_output


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from htmresearch.support.sequence_learning_utils import *
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['pdf.fonttype'] = 42

plt.ion()


DATA_DIR = "./data"
MODEL_PARAMS_DIR = "./model_params"


def getMetricSpecs(predictedField, stepsAhead=5):
  _METRIC_SPECS = (
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'negativeLogLikelihood',
                         'window': 1000, 'steps': stepsAhead}),
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'nrmse', 'window': 1000,
                         'steps': stepsAhead}),
  )
  return _METRIC_SPECS


def createModel(modelParams):
  model = ModelFactory.create(modelParams)
  model.enableInference({"predictedField": predictedField})
  return model


def getModelParamsFromName(dataSet):
  importName = "model_params.%s_model_params" % (
    dataSet.replace(" ", "_").replace("-", "_")
  )
  print "Importing model params from %s" % importName
  try:
    importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'. Run swarm first!"
                    % dataSet)
  return importedModelParams


def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nCompare TM performance with trivial predictor using "
                              "model outputs in prediction directory "
                              "and outputting results to result directory.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='nyc_taxi',
                    dest="dataSet",
                    help="DataSet Name, choose from rec-center-hourly, nyc_taxi")

  parser.add_option("-p",
                    "--plot",
                    default=False,
                    dest="plot",
                    help="Set to True to plot result")

  parser.add_option("--stepsAhead",
                    help="How many steps ahead to predict. [default: %default]",
                    default=5,
                    type=int)

  parser.add_option("-c",
                    "--classifier",
                    type=str,
                    default='SDRClassifierRegion',
                    dest="classifier",
                    help="Classifier Type: SDRClassifierRegion or CLAClassifierRegion")

  (options, remainder) = parser.parse_args()
  print options

  return options, remainder


def getInputRecord(df, predictedField, i):
  inputRecord = {
    predictedField: float(df[predictedField][i]),
    "timeofday": float(df["timeofday"][i]),
    "dayofweek": float(df["dayofweek"][i]),
  }
  return inputRecord


def printTPRegionParams(tpregion):
  """
  Note: assumes we are using TemporalMemory/TPShim in the TPRegion
  """
  tm = tpregion.getSelf()._tfdr
  print "------------PY  TemporalMemory Parameters ------------------"
  print "numberOfCols             =", tm.getColumnDimensions()
  print "cellsPerColumn           =", tm.getCellsPerColumn()
  print "minThreshold             =", tm.getMinThreshold()
  print "activationThreshold      =", tm.getActivationThreshold()
  print "newSynapseCount          =", tm.getMaxNewSynapseCount()
  print "initialPerm              =", tm.getInitialPermanence()
  print "connectedPerm            =", tm.getConnectedPermanence()
  print "permanenceInc            =", tm.getPermanenceIncrement()
  print "permanenceDec            =", tm.getPermanenceDecrement()
  print "predictedSegmentDecrement=", tm.getPredictedSegmentDecrement()
  print



def runMultiplePass(df, model, nMultiplePass, nTrain):
  """
  run CLA model through data record 0:nTrain nMultiplePass passes
  """
  predictedField = model.getInferenceArgs()['predictedField']
  print "run TM through the train data multiple times"
  for nPass in xrange(nMultiplePass):
    for j in xrange(nTrain):
      inputRecord = getInputRecord(df, predictedField, j)
      result = model.run(inputRecord)
      if j % 100 == 0:
        print " pass %i, record %i" % (nPass, j)
    # reset temporal memory
    model._getTPRegion().getSelf()._tfdr.reset()

  return model



def runMultiplePassSPonly(df, model, nMultiplePass, nTrain):
  """
  run CLA model SP through data record 0:nTrain nMultiplePass passes
  """

  predictedField = model.getInferenceArgs()['predictedField']
  print "run TM through the train data multiple times"
  for nPass in xrange(nMultiplePass):
    for j in xrange(nTrain):
      inputRecord = getInputRecord(df, predictedField, j)
      model._sensorCompute(inputRecord)
      model._spCompute()
      if j % 400 == 0:
        print " pass %i, record %i" % (nPass, j)

  return model



def movingAverage(a, n):
  movingAverage = []

  for i in xrange(len(a)):
    start = max(0, i - n)
    values = a[start:i+1]
    movingAverage.append(sum(values) / float(len(values)))

  return movingAverage



if __name__ == "__main__":
  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  plot = _options.plot
  classifierType = _options.classifier

  if dataSet == "rec-center-hourly":
    DATE_FORMAT = "%m/%d/%y %H:%M" # '7/2/10 0:00'
    predictedField = "kw_energy_consumption"
  elif dataSet == "nyc_taxi" or dataSet == "nyc_taxi_perturb" or dataSet =="nyc_taxi_perturb_baseline":
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    predictedField = "passenger_count"
  else:
    raise RuntimeError("un recognized dataset")

  if dataSet == "nyc_taxi" or dataSet == "nyc_taxi_perturb" or dataSet =="nyc_taxi_perturb_baseline":
    modelParams = getModelParamsFromName("nyc_taxi")
  else:
    modelParams = getModelParamsFromName(dataSet)
  modelParams['modelParams']['clParams']['steps'] = str(_options.stepsAhead)
  modelParams['modelParams']['clParams']['regionName'] = classifierType

  print "Creating model from %s..." % dataSet

  # use customized CLA model
  model = CLAModel_custom(**modelParams['modelParams'])
  model.enableInference({"predictedField": predictedField})
  model.enableLearning()
  model._spLearningEnabled = True
  model._tpLearningEnabled = True

  printTPRegionParams(model._getTPRegion())

  inputData = "%s/%s.csv" % (DATA_DIR, dataSet.replace(" ", "_"))

  sensor = model._getSensorRegion()
  encoderList = sensor.getSelf().encoder.getEncoderList()
  if sensor.getSelf().disabledEncoder is not None:
    classifier_encoder = sensor.getSelf().disabledEncoder.getEncoderList()
    classifier_encoder = classifier_encoder[0]
  else:
    classifier_encoder = None

  _METRIC_SPECS = getMetricSpecs(predictedField, stepsAhead=_options.stepsAhead)
  metric = metrics.getModule(_METRIC_SPECS[0])
  metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                  model.getInferenceType())

  if plot:
    plotCount = 1
    plotHeight = max(plotCount * 3, 6)
    fig = plt.figure(figsize=(14, plotHeight))
    gs = gridspec.GridSpec(plotCount, 1)
    plt.title(predictedField)
    plt.ylabel('Data')
    plt.xlabel('Timed')
    plt.tight_layout()
    plt.ion()

  print "Load dataset: ", dataSet
  df = pd.read_csv(inputData, header=0, skiprows=[1, 2])

  nMultiplePass = 5
  nTrain = 5000
  print " run SP through the first %i samples %i passes " %(nMultiplePass, nTrain)
  model = runMultiplePassSPonly(df, model, nMultiplePass, nTrain)
  model._spLearningEnabled = False

  maxBucket = classifier_encoder.n - classifier_encoder.w + 1
  likelihoodsVecAll = np.zeros((maxBucket, len(df)))

  prediction_nstep = None
  time_step = []
  actual_data = []
  patternNZ_track = []
  predict_data = np.zeros((_options.stepsAhead, 0))
  predict_data_ML = []
  negLL_track = []

  activeCellNum = []
  predCellNum = []
  predSegmentNum = []
  predictedActiveColumnsNum = []
  trueBucketIndex = []
  sp = model._getSPRegion().getSelf()._sfdr
  spActiveCellsCount = np.zeros(sp.getColumnDimensions())

  output = nupic_output.NuPICFileOutput([dataSet])

  for i in xrange(len(df)):
    inputRecord = getInputRecord(df, predictedField, i)
    tp = model._getTPRegion()
    tm = tp.getSelf()._tfdr
    prePredictiveCells = tm.getPredictiveCells()
    prePredictiveColumn = np.array(list(prePredictiveCells)) / tm.cellsPerColumn

    result = model.run(inputRecord)
    trueBucketIndex.append(model._getClassifierInputRecord(inputRecord).bucketIndex)

    predSegmentNum.append(len(tm.activeSegments))

    sp = model._getSPRegion().getSelf()._sfdr
    spOutput = model._getSPRegion().getOutputData('bottomUpOut')
    spActiveCellsCount[spOutput.nonzero()[0]] += 1

    activeDutyCycle = np.zeros(sp.getColumnDimensions(), dtype=np.float32)
    sp.getActiveDutyCycles(activeDutyCycle)
    overlapDutyCycle = np.zeros(sp.getColumnDimensions(), dtype=np.float32)
    sp.getOverlapDutyCycles(overlapDutyCycle)

    if i % 100 == 0 and i > 0:
      plt.figure(1)
      plt.clf()
      plt.subplot(2, 2, 1)
      plt.hist(overlapDutyCycle)
      plt.xlabel('overlapDutyCycle')

      plt.subplot(2, 2, 2)
      plt.hist(activeDutyCycle)
      plt.xlabel('activeDutyCycle-1000')

      plt.subplot(2, 2, 3)
      plt.hist(spActiveCellsCount)
      plt.xlabel('activeDutyCycle-Total')
      plt.draw()

    tp = model._getTPRegion()
    tm = tp.getSelf()._tfdr
    tpOutput = tm.infActiveState['t']

    predictiveCells = tm.getPredictiveCells()
    predCellNum.append(len(predictiveCells))
    predColumn = np.array(list(predictiveCells))/ tm.cellsPerColumn

    patternNZ = tpOutput.reshape(-1).nonzero()[0]
    activeColumn = patternNZ / tm.cellsPerColumn
    activeCellNum.append(len(patternNZ))

    predictedActiveColumns = np.intersect1d(prePredictiveColumn, activeColumn)
    predictedActiveColumnsNum.append(len(predictedActiveColumns))

    result.metrics = metricsManager.update(result)

    negLL = result.metrics["multiStepBestPredictions:multiStep:"
               "errorMetric='negativeLogLikelihood':steps=%d:window=1000:"
               "field=%s"%(_options.stepsAhead, predictedField)]
    if i % 100 == 0 and i>0:
      negLL = result.metrics["multiStepBestPredictions:multiStep:"
               "errorMetric='negativeLogLikelihood':steps=%d:window=1000:"
               "field=%s"%(_options.stepsAhead, predictedField)]
      nrmse = result.metrics["multiStepBestPredictions:multiStep:"
               "errorMetric='nrmse':steps=%d:window=1000:"
               "field=%s"%(_options.stepsAhead, predictedField)]

      numActiveCell = np.mean(activeCellNum[-100:])
      numPredictiveCells = np.mean(predCellNum[-100:])
      numCorrectPredicted = np.mean(predictedActiveColumnsNum[-100:])

      print "After %i records, %d-step negLL=%f nrmse=%f ActiveCell %f PredCol %f CorrectPredCol %f" % \
            (i, _options.stepsAhead, negLL, nrmse, numActiveCell,
             numPredictiveCells, numCorrectPredicted)

    last_prediction = prediction_nstep
    prediction_nstep = \
      result.inferences["multiStepBestPredictions"][_options.stepsAhead]
    output.write([i], [inputRecord[predictedField]], [float(prediction_nstep)])

    bucketLL = \
      result.inferences['multiStepBucketLikelihoods'][_options.stepsAhead]
    likelihoodsVec = np.zeros((maxBucket,))
    if bucketLL is not None:
      for (k, v) in bucketLL.items():
        likelihoodsVec[k] = v

    time_step.append(i)
    actual_data.append(inputRecord[predictedField])
    predict_data_ML.append(
      result.inferences['multiStepBestPredictions'][_options.stepsAhead])
    negLL_track.append(negLL)

    likelihoodsVecAll[0:len(likelihoodsVec), i] = likelihoodsVec

    if plot and i > 500:
      # prepare data for display
      if i > 100:
        time_step_display = time_step[-500:-_options.stepsAhead]
        actual_data_display = actual_data[-500+_options.stepsAhead:]
        predict_data_ML_display = predict_data_ML[-500:-_options.stepsAhead]
        likelihood_display = likelihoodsVecAll[:, i-499:i-_options.stepsAhead+1]
        xl = [(i)-500, (i)]
      else:
        time_step_display = time_step
        actual_data_display = actual_data
        predict_data_ML_display = predict_data_ML
        likelihood_display = likelihoodsVecAll[:, :i+1]
        xl = [0, (i)]

      plt.figure(2)
      plt.clf()
      plt.imshow(likelihood_display,
                 extent=(time_step_display[0], time_step_display[-1], 0, 40000),
                 interpolation='nearest', aspect='auto',
                 origin='lower', cmap='Reds')
      plt.colorbar()
      plt.plot(time_step_display, actual_data_display, 'k', label='Data')
      plt.plot(time_step_display, predict_data_ML_display, 'b', label='Best Prediction')
      plt.xlim(xl)
      plt.xlabel('Time')
      plt.ylabel('Prediction')
      # plt.title('TM, useTimeOfDay='+str(True)+' '+dataSet+' test neg LL = '+str(np.nanmean(negLL)))
      plt.xlim([17020, 17300])
      plt.ylim([0, 30000])
      plt.clim([0, 1])
      plt.draw()

  predData_TM_n_step = np.roll(np.array(predict_data_ML), _options.stepsAhead)
  nTest = len(actual_data) - nTrain - _options.stepsAhead
  NRMSE_TM = NRMSE(actual_data[nTrain:nTrain+nTest], predData_TM_n_step[nTrain:nTrain+nTest])
  print "NRMSE on test data: ", NRMSE_TM
  output.close()


  # calculate neg-likelihood
  predictions = np.transpose(likelihoodsVecAll)
  truth = np.roll(actual_data, -5)

  from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
  encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
  from plot import computeLikelihood, plotAccuracy

  bucketIndex2 = []
  negLL  = []
  minProb = 0.0001
  for i in xrange(len(truth)):
    bucketIndex2.append(np.where(encoder.encode(truth[i]))[0])
    outOfBucketProb = 1 - sum(predictions[i,:])
    prob = predictions[i, bucketIndex2[i]]
    if prob == 0:
      prob = outOfBucketProb
    if prob < minProb:
      prob = minProb
    negLL.append( -np.log(prob))

  negLL = computeLikelihood(predictions, truth, encoder)
  negLL[:5000] = np.nan
  x = range(len(negLL))
  plt.figure()
  plotAccuracy((negLL, x), truth, window=480, errorType='negLL')

  np.save('./result/'+dataSet+classifierType+'TMprediction.npy', predictions)
  np.save('./result/'+dataSet+classifierType+'TMtruth.npy', truth)

  plt.figure()
  activeCellNumAvg = movingAverage(activeCellNum, 100)
  plt.plot(np.array(activeCellNumAvg)/tm.numberOfCells())
  plt.xlabel('data records')
  plt.ylabel('sparsity')
  plt.xlim([0, 5000])

  plt.savefig('result/sparsity_over_training.pdf')

  plt.figure()
  predCellNumAvg = movingAverage(predCellNum, 100)
  predSegmentNumAvg = movingAverage(predSegmentNum, 100)
  # plt.plot(np.array(predCellNumAvg))
  plt.plot(np.array(predSegmentNumAvg),'r', label='NMDA spike')
  plt.plot(activeCellNumAvg,'b', label='spikes')
  plt.xlabel('data records')
  plt.ylabel('NMDA spike #')
  plt.legend()
  plt.xlim([0, 5000])
  plt.ylim([0, 42])
  plt.savefig('result/nmda_spike_over_training.pdf')