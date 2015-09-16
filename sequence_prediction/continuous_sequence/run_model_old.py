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

import os, sys, csv
import pprint
import importlib

from optparse import OptionParser
from nupic.swarming import permutations_runner
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.data.inference_shifter import InferenceShifter

import matplotlib

import datetime
from nupic.frameworks.opf.clamodel import CLAModel
from nupic.algorithms.CLAClassifier import BitHistory
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from swarm_runner import SwarmRunner

# from SWARM_CONFIG import SWARM_CONFIG
import numpy
import pandas as pd
import numpy as np

from nupic.frameworks.opf.opfutils import (InferenceType,
                      InferenceElement,
                      SensorInput,
                      initLogger)


from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf import metrics

DATA_DIR = "./data"
MODEL_PARAMS_DIR = "./model_params"

def getMetricSpecs(predictedField):
  _METRIC_SPECS = (
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'negativeLogLikelihood', 'window': 1000, 'steps': 5}),
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'nrmse', 'window': 1000, 'steps': 5}),
  )
  return _METRIC_SPECS


def getModelParamsFromName(modelName):
  importName = "model_params.%s_model_params" % (
    modelName.replace(" ", "_").replace("-", "_")
  )
  print "Importing model params from %s" % importName
  try:
    importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'. Run swarm first!"
                    % modelName)

  return importedModelParams


def createModel(modelParams):
  model = ModelFactory.create(modelParams)
  model.enableInference({"predictedField": SWARM_CONFIG['inferenceArgs']['predictedField']})
  return model


def runNupicModel(filePath, model, plot, useDeltaEncoder=True, savePrediction=True):

  fileName = os.path.splitext(os.path.basename(filePath))[0]

  inputField = SWARM_CONFIG["includedFields"][0]['fieldName']
  predictedField = SWARM_CONFIG['inferenceArgs']['predictedField']
  predictionSteps = SWARM_CONFIG['inferenceArgs']['predictionSteps']
  nPredictionSteps = len(predictionSteps)

  print "inputField: ", inputField
  print "predictedField: ", predictedField

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

  if savePrediction:
    outputFileName = './prediction/'+fileName+'_TM_pred.csv'
    outputFile = open(outputFileName,"w")
    csvWriter = csv.writer(outputFile)
    csvWriter.writerow(['step', 'data','prediction'])
    csvWriter.writerow(['int', 'float','float'])
    csvWriter.writerow(['', ''])

  data = pd.read_csv(filePath, header=0, skiprows=[1,2])

  predictedFieldVals = data[predictedField].astype('float')
  if useDeltaEncoder:
    firstDifference = predictedFieldVals.diff()

  time_step = []
  actual_data = []
  predict_data = np.zeros((nPredictionSteps, 0))
  for i in xrange(len(data)):
    time_step.append(i)
    if (i % 100 == 0):
      print "Read %i lines..." % i

    inputRecord = {}
    for field in range(len(SWARM_CONFIG["includedFields"])):
      fieldName = SWARM_CONFIG["includedFields"][field]['fieldName']
      inputRecord[fieldName] = float(data[fieldName].values[i])

    if useDeltaEncoder:
      inputRecord[predictedField] = float(firstDifference.values[i])

    result = model.run(inputRecord)

    actual_data.append(float(predictedFieldVals.values[i]))
    prediction = result.inferences["multiStepBestPredictions"]
    prediction_values = np.array(prediction.values()).reshape((nPredictionSteps, 1))
    prediction_values = np.where(prediction_values == np.array(None), 0, prediction_values)

    if useDeltaEncoder:
      prediction_values += float(predictedFieldVals.values[i])

    predict_data = np.concatenate((predict_data, prediction_values),1)

    if plot:
      if len(actual_data) > 100:
        time_step_display = time_step[-100:]
        actual_data_display = actual_data[-100:]
        predict_data_display = predict_data[-1,-100:]
        xl = [len(actual_data)-100, len(actual_data)]
      else:
        time_step_display = time_step
        actual_data_display = actual_data
        predict_data_display = predict_data[-1,:]
        xl = [0, len(actual_data)]

      plt.plot(time_step_display, actual_data_display,'k')
      plt.plot(time_step_display, predict_data_display,'r')
      plt.xlim(xl)
      plt.draw()

    allPrediction = list(prediction_values.reshape(nPredictionSteps,))

    if savePrediction:
      csvWriter.writerow([time_step[-1], actual_data[-1], allPrediction[0]])

  if savePrediction:
    outputFile.close()


def calculateFirstDifference(filePath, outputFilePath):
  """
  Create an auxiliary data file that contains first difference of the predicted field
  :param filePath: path of the original data file
  """
  predictedField = SWARM_CONFIG['inferenceArgs']['predictedField']

  data = pd.read_csv(filePath, header=0, skiprows=[1,2])
  predictedFieldVals = data[predictedField].astype('float')
  firstDifference = predictedFieldVals.diff()
  data[predictedField] = firstDifference

  inputFile = open(filePath, "r")
  outputFile = open(outputFilePath, "w")
  csvReader = csv.reader(inputFile)
  csvWriter = csv.writer(outputFile)
  # write headlines
  for _ in xrange(3):
    readrow = csvReader.next()
    csvWriter.writerow(readrow)
  for i in xrange(len(data)):
    csvWriter.writerow(list(data.iloc[i]))

  inputFile.close()
  outputFile.close()


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

  parser.add_option("-f",
                    "--useDeltaEncoder",
                    default=False,
                    dest="useDeltaEncoder",
                    help="Set to True to use delta encoder")

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


def runExperiment(SWARM_CONFIG, useDeltaEncoder=False):

  filePath = SWARM_CONFIG["streamDef"]['streams'][0]['source']
  filePath = filePath[7:]
  fileName = os.path.splitext(filePath)[0]

  # calculate first difference if delta encoder is used
  if useDeltaEncoder:
    filePathtrain = fileName + '_FirstDifference' + '.csv'
    calculateFirstDifference(filePath, filePathtrain)

    filePathtestOriginal = fileName+'_cont'+'.csv'
    filePathtest = fileName + '_FirstDifference' +'_cont'+'.csv'
    calculateFirstDifference(filePathtestOriginal, filePathtest)
  else:
    filePathtrain = fileName + '.csv'
    filePathtest = fileName + '_cont'+'.csv'
    filePathtestOriginal = filePathtest

  modelName = os.path.splitext(os.path.basename(filePathtrain))[0]
  modelParams = getModelParamsFromName(modelName)
  model = createModel(modelParams)

  print 'run model on training data ', filePathtrain
  runNupicModel(filePath, model, plot=False, useDeltaEncoder=useDeltaEncoder, savePrediction=True)

  try:
    print 'run model on test data ', filePathtestOriginal
    runNupicModel(filePathtestOriginal, model, plot=False, useDeltaEncoder=useDeltaEncoder, savePrediction=True)
  except ImportError:
    raise Exception("No continuation file exist at %s " % filePathtestOriginal)


if __name__ == "__main__":
  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  useDeltaEncoder = _options.useDeltaEncoder

  SWARM_CONFIG = SwarmRunner.importSwarmDescription(dataSet)
  # runExperiment(SWARM_CONFIG, useDeltaEncoder)


  filePath = SWARM_CONFIG["streamDef"]['streams'][0]['source']
  filePath = filePath[7:]
  fileName = os.path.splitext(filePath)[0]

  # calculate first difference if delta encoder is used
  if useDeltaEncoder:
    filePathtrain = fileName + '_FirstDifference' + '.csv'
    calculateFirstDifference(filePath, filePathtrain)

    filePathtestOriginal = fileName+'_cont'+'.csv'
    filePathtest = fileName + '_FirstDifference' +'_cont'+'.csv'
    calculateFirstDifference(filePathtestOriginal, filePathtest)
  else:
    filePathtrain = fileName + '.csv'
    filePathtest = fileName + '_cont'+'.csv'
    filePathtestOriginal = filePathtest


  plot = False
  savePrediction = False

  fileName = os.path.splitext(os.path.basename(filePath))[0]

  inputField = SWARM_CONFIG["includedFields"][0]['fieldName']
  predictedField = SWARM_CONFIG['inferenceArgs']['predictedField']
  predictionSteps = SWARM_CONFIG['inferenceArgs']['predictionSteps']
  nPredictionSteps = len(predictionSteps)

  print "inputField: ", inputField
  print "predictedField: ", predictedField
  plt.close('all')
  if plot:
    plotCount = 1
    plotHeight = max(plotCount * 3, 6)
    fig = plt.figure(figsize=(14, plotHeight))
    gs = gridspec.GridSpec(plotCount, 1)
    plt.title(predictedField)
    plt.ylabel('Data')
    plt.xlabel('Timed')
    # plt.tight_layout()
    plt.ion()

  if savePrediction:
    outputFileName = './prediction/'+fileName+'_TM_pred.csv'
    outputFile = open(outputFileName,"w")
    csvWriter = csv.writer(outputFile)
    csvWriter.writerow(['step', 'data','prediction'])
    csvWriter.writerow(['int', 'float','float'])
    csvWriter.writerow(['', ''])

  data = pd.read_csv(filePath, header=0, skiprows=[1,2])

  predictedFieldVals = data[predictedField].astype('float')
  if useDeltaEncoder:
    firstDifference = predictedFieldVals.diff()


  modelName = os.path.splitext(os.path.basename(filePathtrain))[0]
  modelParams = getModelParamsFromName(modelName)

  model = createModel(modelParams)

  model.enableLearning()

  # from nupic.frameworks.opf.clamodel import CLAModel
  # model = CLAModel(**modelParams['modelParams'])

  _METRIC_SPECS = getMetricSpecs(predictedField)
  metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                  model.getInferenceType())


  # from clamodel_custom import CLAModel_custom
  # model = CLAModel_custom(True, True, **modelParams['modelParams'])

  # # pre-run -> sp only
  # for i in xrange(len(data)):
  #   inputRecord = {}
  #   for field in range(len(SWARM_CONFIG["includedFields"])):
  #     fieldName = SWARM_CONFIG["includedFields"][field]['fieldName']
  #     inputRecord[fieldName] = float(data[fieldName].values[i])
  #
  #   if useDeltaEncoder:
  #     inputRecord[predictedField] = float(firstDifference.values[i])
  #
  #   # result = model.run(inputRecord)
  #
  #   results = super(CLAModel, model).run(inputRecord)
  #
  #   # code in clamodel.compute
  #   # model._getTPRegion().getSelf().learningMode = 0
  #   model._sensorCompute(inputRecord)
  #   model._spCompute()

  # model.disableLearning()
  #
  time_step = []
  actual_data = []
  patternNZ_track = []
  predict_data = np.zeros((nPredictionSteps, 0))
  predict_data_ML = np.zeros((nPredictionSteps, 0))

  sensor = model._getSensorRegion()
  encoderList = sensor.getSelf().encoder.getEncoderList()
  if sensor.getSelf().disabledEncoder is not None:
    encoderList = sensor.getSelf().disabledEncoder.getEncoderList()

  encoder = encoderList[0]
  # encoder = model._classifierInputEncoder
  maxBucket = encoder.n - encoder.w + 1
  likelihoodsVecAll = np.zeros((maxBucket, len(data)))


  #
  inputData = "%s/%s.csv" % (DATA_DIR, dataSet.replace(" ", "_"))
  print "Load dataset: ", dataSet
  df = pd.read_csv(inputData, header=0, skiprows=[1, 2])

  for i in xrange(len(data)):
    time_step.append(i)
    if (i % 100 == 0):
      print "Read %i lines..." % i

    recordNum = i
    inputRecord = getInputRecord(df, predictedField, i)
    # inputRecord = {}
    # for field in range(len(SWARM_CONFIG["includedFields"])):
    #   fieldName = SWARM_CONFIG["includedFields"][field]['fieldName']
    #   inputRecord[fieldName] = float(data[fieldName].values[i])
    #
    # if useDeltaEncoder:
    #   inputRecord[predictedField] = float(firstDifference.values[i])

    results = model.run(inputRecord)

    # details within model.run

    results = super(CLAModel, model).run(inputRecord)

    # code in clamodel.compute
    # model._getTPRegion().getSelf().learningMode = 0
    model._sensorCompute(inputRecord)
    model._spCompute()
    model._tpCompute()

    results.sensorInput = model._getSensorInputRecord(inputRecord)

    results.inferences = {}
    inferences = {}

    tp = model._getTPRegion()
    tpOutput = tp.getSelf()._tfdr.infActiveState['t']
    patternNZ = tpOutput.reshape(-1).nonzero()[0]

    # sp = model._getSPRegion()
    # spOutput = sp.getOutputData('bottomUpOut')
    # patternNZ = spOutput.nonzero()[0]

    # print "  current input: ", inputRecord[predictedField]
    # print "  patternNZ (%d):" % len(patternNZ)

    rawInput = inputRecord
    inputTSRecordIdx = rawInput.get('_timestampRecordIdx')
    inferences = model._handleCLAClassifierMultiStep(
                                        patternNZ=patternNZ,
                                        inputTSRecordIdx=inputTSRecordIdx,
                                        rawInput=rawInput)

    # # originally in clamodel._handleCLAClassifierMultiStep
    # rawInput = inputRecord
    # inferenceArgs = model.getInferenceArgs()
    # # predictedFieldName = inferenceArgs.get('predictedField', None)
    # model._predictedFieldName = predictedField
    #
    # classifier = model._getClassifierRegion()
    # sensor = model._getSensorRegion()
    # minLikelihoodThreshold = model._minLikelihoodThreshold
    # maxPredictionsPerStep = model._maxPredictionsPerStep
    # needLearning = model.isLearningEnabled()
    # inferences = {}
    # # set the classifier input encoder
    # fieldNames = sensor.getSelf().encoder.getScalarNames()
    # model._predictedFieldIdx = fieldNames.index(predictedField)
    # encoderList = sensor.getSelf().encoder.getEncoderList()
    # model._classifierInputEncoder = encoderList[model._predictedFieldIdx]
    #
    # absoluteValue = rawInput[predictedField]
    # bucketIdx = model._classifierInputEncoder.getBucketIndices(absoluteValue)[0]
    #
    # actualValue = absoluteValue
    #
    # classifier.setParameter('inferenceMode', True)
    # classifier.setParameter('learningMode', needLearning)
    # classification = {'bucketIdx': bucketIdx,
    #                   'actValue': actualValue}
    #
    #
    # # code in CLAClassifier.compute
    # classifier = model._getClassifierRegion()
    # learn = True
    # # learn = classifier.getSelf().learningMode
    # infer = classifier.getSelf().inferenceMode
    # cl = classifier.getSelf()._claClassifier
    #
    # cl._recordNumMinusLearnIteration = 0
    # # Save the offset between recordNum and learnIteration if this is the first
    # #  compute
    # if cl._recordNumMinusLearnIteration is None:
    #   cl._recordNumMinusLearnIteration = recordNum - cl._learnIteration
    #
    # # Update the learn iteration
    # cl._learnIteration = recordNum - cl._recordNumMinusLearnIteration
    #
    # cl.verbosity = 0
    # if cl.verbosity >= 1:
    #   print
    #   print "  recordNum:", recordNum
    #   print "  learnIteration:", cl._learnIteration
    #   print "  current input: ", inputRecord[predictedField]
    #   print "  patternNZ (%d):" % len(patternNZ), patternNZ
    #   # print "  classificationIn:", classification
    #
    # # Store pattern in our history
    # cl._patternNZHistory.append((cl._learnIteration, patternNZ))
    # patternNZ_track.append(patternNZ)
    #
    # # stepsInHistory = []
    # # for _ in xrange(len(cl._patternNZHistory)):
    # #   stepsInHistory.append(cl._patternNZHistory[_][0])
    # # print "stepsInHistory: ", stepsInHistory
    #
    # # ------------------------------------------------------------------------
    # # Inference:
    # # For each active bit in the activationPattern, get the classification
    # # votes
    # if infer:
    #   # Return value dict. For buckets which we don't have an actual value
    #   # for yet, just plug in any valid actual value. It doesn't matter what
    #   # we use because that bucket won't have non-zero likelihood anyways.
    #
    #   # NOTE: If doing 0-step prediction, we shouldn't use any knowledge
    #   #  of the classification input during inference.
    #   if cl.steps[0] == 0:
    #     defaultValue = 0
    #   else:
    #     defaultValue = classification["actValue"]
    #   actValues = [x if x is not None else defaultValue
    #                for x in cl._actualValues]
    #   retval = {"actualValues": actValues}
    #
    #   # For each n-step prediction...
    #   for nSteps in cl.steps:
    #
    #     # Accumulate bucket index votes and actValues into these arrays
    #     sumVotes = numpy.zeros(cl._maxBucketIdx+1)
    #     bitVotes = numpy.zeros(cl._maxBucketIdx+1)
    #
    #     # For each active bit, get the votes
    #     validBit = []
    #     for bit in patternNZ:
    #       key = (bit, nSteps)
    #       history = cl._activeBitHistory.get(key, None)
    #       if history is None:
    #         continue
    #
    #       validBit.append(bit)
    #       bitVotes.fill(0)
    #       history.infer(votes=bitVotes)
    #
    #       sumVotes += bitVotes
    #
    #     # Return the votes for each bucket, normalized
    #     total = sumVotes.sum()
    #     # print " validBit ", validBit
    #     # print " total vote: ", total
    #     if total > 0:
    #       sumVotes /= total
    #     else:
    #       # If all buckets have zero probability then simply make all of the
    #       # buckets equally likely. There is no actual prediction for this
    #       # timestep so any of the possible predictions are just as good.
    #       if sumVotes.size > 0:
    #         sumVotes = numpy.ones(sumVotes.shape)
    #         sumVotes /= sumVotes.size
    #
    #     retval[nSteps] = sumVotes
    #
    #
    # for bit in patternNZ:
    #   key = (bit, nSteps)
    #   history = cl._activeBitHistory.get(key, None)
    #
    # # ------------------------------------------------------------------------
    # # Learning:
    # # For each active bit in the activationPattern, store the classification
    # # info. If the bucketIdx is None, we can't learn. This can happen when the
    # # field is missing in a specific record.
    # if learn and classification["bucketIdx"] is not None:
    #
    #   # Get classification info
    #   bucketIdx = classification["bucketIdx"]
    #   actValue = classification["actValue"]
    #
    #   # Update maxBucketIndex
    #   cl._maxBucketIdx = max(cl._maxBucketIdx, bucketIdx)
    #
    #   # Update rolling average of actual values if it's a scalar. If it's
    #   # not, it must be a category, in which case each bucket only ever
    #   # sees one category so we don't need a running average.
    #   while cl._maxBucketIdx > len(cl._actualValues)-1:
    #     cl._actualValues.append(None)
    #   if cl._actualValues[bucketIdx] is None:
    #     cl._actualValues[bucketIdx] = actValue
    #   else:
    #     if isinstance(actValue, int) or isinstance(actValue, float):
    #       cl._actualValues[bucketIdx] = \
    #               (1.0 - cl.actValueAlpha) * cl._actualValues[bucketIdx] \
    #                + cl.actValueAlpha * actValue
    #     else:
    #       cl._actualValues[bucketIdx] = actValue
    #
    #   # Train each pattern that we have in our history that aligns with the
    #   # steps we have in cl.steps
    #   for nSteps in cl.steps:
    #
    #     # Do we have the pattern that should be assigned to this classification
    #     # in our pattern history? If not, skip it
    #     found = False
    #     for (iteration, learnPatternNZ) in cl._patternNZHistory:
    #       if iteration == cl._learnIteration - nSteps:
    #         found = True
    #         break
    #     if not found:
    #       continue
    #
    #     # Store classification info for each active bit from the pattern
    #     # that we got nSteps time steps ago.
    #     for bit in learnPatternNZ:
    #
    #       # Get the history structure for this bit and step #
    #       key = (bit, nSteps)
    #       history = cl._activeBitHistory.get(key, None)
    #       if history is None:
    #         history = cl._activeBitHistory[key] = BitHistory(cl,
    #                     bitNum=bit, nSteps=nSteps)
    #
    #       # Store new sample
    #       history.store(iteration=cl._learnIteration,
    #                     bucketIdx=bucketIdx)
    #
    # # ------------------------------------------------------------------------
    # # Verbose print
    # if infer and cl.verbosity >= 2:
    #   print "  inference: combined bucket likelihoods:"
    #   print "    actual bucket values:", retval["actualValues"]
    #   for (nSteps, votes) in retval.items():
    #     if nSteps == "actualValues":
    #       continue
    #     # print "    %d steps: " % (nSteps), _pFormatArray(votes)
    #     bestBucketIdx = votes.argmax()
    #     print "      most likely bucket idx: %d, value: %s" % (bestBucketIdx,
    #                         retval["actualValues"][bestBucketIdx])
    #   print
    #
    #
    # # back to _handleCLAClassifierMultiStep
    # classifier.getSelf()._claClassifier = cl
    # clResults = retval
    #
    #
    # # ---------------------------------------------------------------
    # # Get the prediction for every step ahead learned by the classifier
    # predictionSteps = classifier.getParameter('steps')
    # predictionSteps = [int(x) for x in predictionSteps.split(',')]
    #
    # # We will return the results in this dict. The top level keys
    # # are the step number, the values are the relative likelihoods for
    # # each classification value in that time step, represented as
    # # another dict where the keys are the classification values and
    # # the values are the relative likelihoods.
    # inferences[InferenceElement.multiStepPredictions] = dict()
    # inferences[InferenceElement.multiStepBestPredictions] = dict()
    # inferences[InferenceElement.multiStepBucketLikelihoods] = dict()
    #
    # # ======================================================================
    # # Plug in the predictions for each requested time step.
    # for steps in predictionSteps:
    #   # From the clResults, compute the predicted actual value. The
    #   # CLAClassifier classifies the bucket index and returns a list of
    #   # relative likelihoods for each bucket. Let's find the max one
    #   # and then look up the actual value from that bucket index
    #   likelihoodsVec = clResults[steps]
    #   bucketValues = clResults['actualValues']
    #
    #   # Create a dict of value:likelihood pairs. We can't simply use
    #   #  dict(zip(bucketValues, likelihoodsVec)) because there might be
    #   #  duplicate bucketValues (this happens early on in the model when
    #   #  it doesn't have actual values for each bucket so it returns
    #   #  multiple buckets with the same default actual value).
    #   likelihoodsDict = dict()
    #   bestActValue = None
    #   bestProb = None
    #   for (actValue, prob) in zip(bucketValues, likelihoodsVec):
    #     if actValue in likelihoodsDict:
    #       likelihoodsDict[actValue] += prob
    #     else:
    #       likelihoodsDict[actValue] = prob
    #     # Keep track of best
    #     if bestProb is None or likelihoodsDict[actValue] > bestProb:
    #       bestProb = likelihoodsDict[actValue]
    #       bestActValue = actValue
    #
    #
    #   # Remove entries with 0 likelihood or likelihood less than
    #   # minLikelihoodThreshold, but don't leave an empty dict.
    #   likelihoodsDict = CLAModel._removeUnlikelyPredictions(
    #       likelihoodsDict, minLikelihoodThreshold, maxPredictionsPerStep)
    #
    #   bucketLikelihood = {}
    #   for k in likelihoodsDict.keys():
    #     bucketLikelihood[model._classifierInputEncoder.getBucketIndices(k)[0]] = likelihoodsDict[k]
    #
    #   # ---------------------------------------------------------------------
    #   # Normal case, no delta encoder. Just plug in all our multi-step predictions
    #   #  with likelihoods as well as our best prediction
    #
    #   # The multiStepPredictions element holds the probabilities for each
    #   #  bucket
    #   inferences[InferenceElement.multiStepPredictions][steps] = \
    #                                                 likelihoodsDict
    #   inferences[InferenceElement.multiStepBestPredictions][steps] = \
    #                                           bestActValue
    #   inferences[InferenceElement.multiStepBucketLikelihoods][steps] = bucketLikelihood
    #

    results.inferences.update(inferences)

    results.predictedFieldIdx = model._predictedFieldIdx
    results.predictedFieldName = model._predictedFieldName
    results.classifierInput = model._getClassifierInputRecord(inputRecord)

    # calculate metric
    results.metrics = metricsManager.update(results)

    counter = i
    if counter % 100 == 0:
      negLL = results.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='negativeLogLikelihood':steps=5:window=1000:"
                             "field="+predictedField]
      nrmse = results.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='nrmse':steps=5:window=1000:"
                             "field="+predictedField]
      print "After %i records, 5-step negLL=%f nrmse=%f" % (counter, negLL, nrmse)

    # print "results (current): ", results.inferences
    continue

    bucketLL = results.inferences['multiStepBucketLikelihoods'][5]
    likelihoodsVec = np.zeros((maxBucket,))
    for (k, v) in bucketLL.items():
      likelihoodsVec[k] = v
    likelihoodsVecAll[0:len(likelihoodsVec), i] = likelihoodsVec



    actual_data.append(float(predictedFieldVals.values[i]))

    prediction = results.inferences['multiStepPredictions'][5]
    predicted_values = sorted(prediction.keys())
    probability = np.zeros(len(predicted_values))
    for i in xrange(len(predicted_values)):
      probability[i] = prediction[predicted_values[i]]
    probability = probability/sum(probability)
    prediction_values = np.dot(probability, predicted_values)
    prediction_values = np.array(prediction_values).reshape((1,1))


    # most likely outcome
    prediction = results.inferences["multiStepBestPredictions"]
    prediction_values_ML = np.array(prediction.values()).reshape((nPredictionSteps, 1))
    prediction_values_ML = np.where(prediction_values_ML == np.array(None), 0, prediction_values_ML)

    if useDeltaEncoder:
      prediction_values += float(predictedFieldVals.values[i])
      prediction_values_ML  += float(predictedFieldVals.values[i])

    predict_data = np.concatenate((predict_data, prediction_values),1)
    predict_data_ML = np.concatenate((predict_data_ML, prediction_values_ML),1)

    negLL = results.metrics.values()[0]
    if plot:
      if len(actual_data) > 100:
        time_step_display = time_step[-100:]
        actual_data_display = actual_data[-100:]
        predict_data_display = predict_data[-1,-100:]
        predict_data_ML_display = predict_data_ML[-1,-100:]
        likelihood_display = likelihoodsVecAll[:, recordNum-100:recordNum]
        xl = [len(actual_data)-100, len(actual_data)]
      else:
        time_step_display = time_step
        actual_data_display = actual_data
        predict_data_display = predict_data[-1,:]
        predict_data_ML_display = predict_data_ML[-1,:]
        likelihood_display = likelihoodsVecAll[:, :recordNum]
        xl = [0, len(actual_data)]


      plt.figure(1)
      plt.clf()
      plt.imshow(likelihood_display, extent=(time_step_display[0], time_step_display[-1], -1, 1),
                 interpolation='nearest', aspect='auto', origin='lower', cmap='Reds')
      plt.plot(time_step_display, actual_data_display, 'k', label='Data')
      plt.plot(time_step_display, predict_data_display, 'r', label='Mean Prediction')
      plt.plot(time_step_display, predict_data_ML_display, 'b', label='Best Prediction')
      plt.xlim(xl)
      plt.title('negLL='+str(negLL))
      plt.draw()

      # plt.legend()

      # if recordNum > 160:
      #   plt.figure(3)
      #   plt.clf()
      #   plt.plot(cl._actualValues, likelihoodsVec)
      #   yl = plt.ylim()
      #   plt.draw()
      #   plt.vlines(inputRecord[predictedField], yl[0], yl[1])
      #   plt.xlabel(' Possible Values ')
      #   plt.ylabel(' Likelihood ')
      #
      #   fileName = './result/'+dataSet+"/predict_distribution"+str(recordNum)+".pdf"
      #   print "save example prediction trace to ", fileName
      #   plt.savefig(fileName)

      # plt.figure(2)
      # plt.clf()
      # plt.plot(predicted_values, probability)


    allPrediction = list(prediction_values.reshape(nPredictionSteps,))

    if savePrediction:
      csvWriter.writerow([time_step[-1], actual_data[-1], allPrediction[0]])

  if savePrediction:
    outputFile.close()
