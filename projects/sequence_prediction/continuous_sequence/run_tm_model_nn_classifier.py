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


import csv
import importlib
from optparse import OptionParser

import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd

from errorMetrics import *
from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager
from nupic.frameworks.opf import metrics
from htmresearch.frameworks.opf.clamodel_custom import CLAModel_custom
import nupic_output
from htmresearch.algorithms.neural_net_classifier import NeuralNetClassifier
from plot import computeLikelihood, plotAccuracy

rcParams.update({'figure.autolayout': True})

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
  print "numberOfCols             =", tm.columnDimensions
  print "cellsPerColumn           =", tm.cellsPerColumn
  print "minThreshold             =", tm.minThreshold
  print "activationThreshold      =", tm.activationThreshold
  print "newSynapseCount          =", tm.maxNewSynapseCount
  print "initialPerm              =", tm.initialPermanence
  print "connectedPerm            =", tm.connectedPermanence
  print "permanenceInc            =", tm.permanenceIncrement
  print "permanenceDec            =", tm.permanenceDecrement
  print "predictedSegmentDecrement=", tm.predictedSegmentDecrement
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



def saveResultToFile(dataSet, predictedInput, algorithmName):
  inputFileName = 'data/' + dataSet + '.csv'
  inputFile = open(inputFileName, "rb")

  csvReader = csv.reader(inputFile)

  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  outputFileName = './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
  outputFile = open(outputFileName, "w")
  csvWriter = csv.writer(outputFile)
  csvWriter.writerow(
    ['timestamp', 'data', 'prediction'])
  csvWriter.writerow(['datetime', 'float', 'float'])
  csvWriter.writerow(['', '', ''])

  for i in xrange(len(predictedInput)):
    row = csvReader.next()
    csvWriter.writerow([row[0], row[1], predictedInput[i]])

  inputFile.close()
  outputFile.close()



if __name__ == "__main__":

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  plot = _options.plot

  if dataSet == "rec-center-hourly":
    DATE_FORMAT = "%m/%d/%y %H:%M"  # '7/2/10 0:00'
    predictedField = "kw_energy_consumption"
  elif dataSet == "nyc_taxi" or dataSet == "nyc_taxi_perturb" or dataSet == "nyc_taxi_perturb_baseline":
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    predictedField = "passenger_count"
  else:
    raise RuntimeError("un recognized dataset")

  if dataSet == "nyc_taxi" or dataSet == "nyc_taxi_perturb" or dataSet == "nyc_taxi_perturb_baseline":
    modelParams = getModelParamsFromName("nyc_taxi")
  else:
    modelParams = getModelParamsFromName(dataSet)
  modelParams['modelParams']['clParams']['steps'] = str(_options.stepsAhead)

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

  # initialize new classifier
  numTMcells = model._getTPRegion().getSelf()._tfdr.numberOfCells()
  nn_classifier = NeuralNetClassifier(numInputs=numTMcells,
                                      steps=[5], alpha=0.005)

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
  print " run SP through the first %i samples %i passes " % (
  nMultiplePass, nTrain)
  model = runMultiplePassSPonly(df, model, nMultiplePass, nTrain)
  model._spLearningEnabled = False

  maxBucket = classifier_encoder.n - classifier_encoder.w + 1
  likelihoodsVecAll = np.zeros((maxBucket, len(df)))
  likelihoodsVecAllNN = np.zeros((maxBucket, len(df)))

  predictionNstep = None
  timeStep = []
  actualData = []
  patternNZTrack = []
  predictData = np.zeros((_options.stepsAhead, 0))
  predictDataCLA = []
  predictDataNN = []
  negLLTrack = []

  activeCellNum = []
  predCellNum = []
  predictedActiveColumnsNum = []
  trueBucketIndex = []
  sp = model._getSPRegion().getSelf()._sfdr
  spActiveCellsCount = np.zeros(sp.getColumnDimensions())

  output = nupic_output.NuPICFileOutput([dataSet])

  for i in xrange(len(df)):
    inputRecord = getInputRecord(df, predictedField, i)
    tp = model._getTPRegion()
    tm = tp.getSelf()._tfdr
    prePredictiveCells = tm.predictiveCells
    prePredictiveColumn = np.array(list(prePredictiveCells)) / tm.cellsPerColumn

    # run model on the input Record
    result = model.run(inputRecord)

    # record and analyze the result
    trueBucketIndex.append(
      model._getClassifierInputRecord(inputRecord).bucketIndex)

    tp = model._getTPRegion()
    tm = tp.getSelf()._tfdr
    tpOutput = tm.infActiveState['t']

    predictiveCells = tm.predictiveCells
    predCellNum.append(len(predictiveCells))
    predColumn = np.array(list(predictiveCells)) / tm.cellsPerColumn

    patternNZ = tpOutput.reshape(-1).nonzero()[0]
    activeColumn = patternNZ / tm.cellsPerColumn
    activeCellNum.append(len(patternNZ))

    predictedActiveColumns = np.intersect1d(prePredictiveColumn, activeColumn)
    predictedActiveColumnsNum.append(len(predictedActiveColumns))

    # fed input to the new classifier
    classification = {'bucketIdx': result.classifierInput.bucketIndex,
                      'actValue': result.classifierInput.dataRow}
    nnRetval = nn_classifier.compute(i, patternNZ, classification,
                                     learn=True, infer=True)
    nnPrediction = nnRetval['actualValues'][np.argmax(nnRetval[5])]
    predictDataNN.append(nnPrediction)
    output.write([i], [inputRecord[predictedField]], [float(nnPrediction)])

    likelihoodsVecAllNN[0:len(nnRetval[5]), i] = nnRetval[5]

    result.metrics = metricsManager.update(result)

    negLL = result.metrics["multiStepBestPredictions:multiStep:"
                           "errorMetric='negativeLogLikelihood':steps=%d:window=1000:"
                           "field=%s" % (_options.stepsAhead, predictedField)]
    if i % 100 == 0 and i > 0:
      negLL = result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='negativeLogLikelihood':steps=%d:window=1000:"
                             "field=%s" % (_options.stepsAhead, predictedField)]
      nrmse = result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='nrmse':steps=%d:window=1000:"
                             "field=%s" % (_options.stepsAhead, predictedField)]

      numActiveCell = np.mean(activeCellNum[-100:])
      numPredictiveCells = np.mean(predCellNum[-100:])
      numCorrectPredicted = np.mean(predictedActiveColumnsNum[-100:])

      print "After %i records, %d-step negLL=%f nrmse=%f ActiveCell %f PredCol %f CorrectPredCol %f" % \
            (i, _options.stepsAhead, negLL, nrmse, numActiveCell,
             numPredictiveCells, numCorrectPredicted)

    bucketLL = \
      result.inferences['multiStepBucketLikelihoods'][_options.stepsAhead]
    likelihoodsVec = np.zeros((maxBucket,))
    if bucketLL is not None:
      for (k, v) in bucketLL.items():
        likelihoodsVec[k] = v

    timeStep.append(i)
    actualData.append(inputRecord[predictedField])
    predictDataCLA.append(
      result.inferences['multiStepBestPredictions'][_options.stepsAhead])
    negLLTrack.append(negLL)

    likelihoodsVecAll[0:len(likelihoodsVec), i] = likelihoodsVec

    if plot and i > 500:
      # prepare data for display
      if i > 100:
        timeStepDisplay = timeStep[-500:-_options.stepsAhead]
        actualDataDisplay = actualData[-500 + _options.stepsAhead:]
        predictDataMLDisplay = predictDataCLA[-500:-_options.stepsAhead]
        predictDataNNDisplay = predictDataNN[-500:-_options.stepsAhead]
        likelihoodDisplay = likelihoodsVecAll[:,
                            i - 499:i - _options.stepsAhead + 1]
        likelihoodDisplayNN = likelihoodsVecAllNN[:,
                              i - 499:i - _options.stepsAhead + 1]
        xl = [(i) - 500, (i)]
      else:
        timeStepDisplay = timeStep
        actualDataDisplay = actualData
        predictDataMLDisplay = predictDataCLA
        predictDataNNDisplay = predictDataNN
        likelihoodDisplayNN = likelihoodsVecAllNN[:, :i + 1]
        likelihoodDisplay = likelihoodsVecAll[:, :i + 1]
        xl = [0, (i)]

      plt.figure(2)
      plt.clf()
      plt.imshow(likelihoodDisplay,
                 extent=(timeStepDisplay[0], timeStepDisplay[-1], 0, 40000),
                 interpolation='nearest', aspect='auto',
                 origin='lower', cmap='Reds')
      plt.plot(timeStepDisplay, actualDataDisplay, 'k', label='Data')
      # plt.plot(timeStepDisplay, predictDataMLDisplay, 'b', label='Best Prediction')
      plt.xlim(xl)
      plt.xlabel('Time')
      plt.ylabel('Prediction')
      plt.title('TM, useTimeOfDay=' + str(
        True) + ' ' + dataSet + ' test neg LL = ' + str(negLL))
      plt.draw()

      plt.figure(3)
      plt.clf()
      plt.imshow(likelihoodDisplayNN,
                 extent=(timeStepDisplay[0], timeStepDisplay[-1], 0, 40000),
                 interpolation='nearest', aspect='auto',
                 origin='lower', cmap='Reds')
      plt.plot(timeStepDisplay, actualDataDisplay, 'k', label='Data')
      # plt.plot(timeStepDisplay, predictDataNNDisplay, 'b', label='Best Prediction')
      plt.xlim(xl)
      plt.xlabel('Time')
      plt.ylabel('Prediction')
      plt.title('TM, useTimeOfDay=' + str(
        True) + ' ' + dataSet + ' test neg LL = ' + str(negLL))
      plt.draw()

  output.close()

  shiftedPredDataCLA = np.roll(np.array(predictDataCLA), _options.stepsAhead)
  shiftedPredDataNN = np.roll(np.array(predictDataNN), _options.stepsAhead)
  nTest = len(actualData) - nTrain - _options.stepsAhead

  NRMSECLA = NRMSE(actualData[nTrain:nTrain + nTest],
                   shiftedPredDataCLA[nTrain:nTrain + nTest])
  NRMSENN = NRMSE(actualData[nTrain:nTrain + nTest],
                  shiftedPredDataNN[nTrain:nTrain + nTest])
  MAPECLA = MAPE(actualData[nTrain:nTrain + nTest],
                    shiftedPredDataCLA[nTrain:nTrain + nTest])
  MAPENN = MAPE(actualData[nTrain:nTrain + nTest],
                   shiftedPredDataNN[nTrain:nTrain + nTest])

  print "NRMSE on test data, CLA: ", NRMSECLA
  print "NRMSE on test data, NN: ", NRMSENN




  # calculate neg-likelihood
  encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
  truth = np.roll(actualData, -5)
  predictions = np.transpose(likelihoodsVecAll)
  negLLCLA = computeLikelihood(predictions, truth, encoder)
  negLLCLA[:5904] = np.nan

  predictions = np.transpose(likelihoodsVecAllNN)
  negLLNN = computeLikelihood(predictions, truth, encoder)
  negLLNN[:5904] = np.nan

  plt.figure()
  plotAccuracy((negLLCLA, range(len(negLLCLA))), truth, window=480, errorType='negLL')
  plotAccuracy((negLLNN, range(len(negLLNN))), truth, window=480, errorType='negLL')

  # np.save('./result/' + dataSet + 'TMprediction.npy', predictions)
  # np.save('./result/' + dataSet + 'TMtruth.npy', truth)


  plt.figure()
  shiftedActualData = np.roll(np.array(actualData), -_options.stepsAhead)
  plt.plot(shiftedActualData)
  plt.plot(predictDataNN)
  plt.plot(predictDataCLA)
  plt.legend(['True', 'NN', 'CLA'])
  plt.xlim([16600, 17000])

  fig, ax = plt.subplots(nrows=1, ncols=3)
  inds = np.arange(2)
  ax1 = ax[0]
  width = 0.5
  ax1.bar(inds, [NRMSECLA,
                 NRMSENN], width=width)
  ax1.set_xticks(inds+width/2)
  ax1.set_ylabel('NRMSE')
  ax1.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax1.set_xticklabels(('CLA', 'NN'))

  ax2 = ax[1]
  ax2.bar(inds, [MAPECLA,
                 MAPENN], width=width)
  ax2.set_xticks(inds+width/2)
  ax2.set_ylabel('MAPE')
  ax2.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax2.set_xticklabels(('CLA', 'NN'))

  ax3 = ax[2]
  ax3.bar(inds, [np.nanmean(negLLCLA),
                 np.nanmean(negLLNN)], width=width)
  ax3.set_xticks(inds+width/2)
  ax3.set_ylabel('negLL')
  ax3.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax3.set_xticklabels(('CLA', 'NN'))