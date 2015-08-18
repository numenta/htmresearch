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

import os, sys, csv
import pprint
import importlib

from optparse import OptionParser
from nupic.swarming import permutations_runner
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.data.inference_shifter import InferenceShifter

import matplotlib

import datetime
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from swarm_runner import SwarmRunner

# from SWARM_CONFIG import SWARM_CONFIG
import pandas as pd
import numpy as np


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
  runExperiment(SWARM_CONFIG, useDeltaEncoder)

