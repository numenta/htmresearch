# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
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

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from SWARM_CONFIG import SWARM_CONFIG
import pandas as pd
import numpy as np

dataSet = 'sine'

def importSwarmDescription(dataSet):
  swarmConfigFileName = 'SWARM_CONFIG_' + dataSet
  try:
    SWARM_CONFIG = importlib.import_module("swarm_description.%s" % swarmConfigFileName).SWARM_CONFIG
  except ImportError:
    raise Exception("No swarm_description exist for '%s'. Create swarm_description first"
                    % dataSet)

  return SWARM_CONFIG

def modelParamsToString(modelParams):
  pp = pprint.PrettyPrinter(indent=2)
  return pp.pformat(modelParams)


def writeModelParamsToFile(modelParams, name):
  cleanName = name.replace(" ", "_").replace("-", "_")
  paramsName = "%s_model_params.py" % cleanName
  outDir = os.path.join(os.getcwd(), 'model_params')
  if not os.path.isdir(outDir):
    os.mkdir(outDir)
  outPath = os.path.join(os.getcwd(), 'model_params', paramsName)
  with open(outPath, "wb") as outFile:
    modelParamsString = modelParamsToString(modelParams)
    outFile.write("MODEL_PARAMS = \\\n%s" % modelParamsString)
  return outPath


def swarmForBestModelParams(swarmConfig, name, maxWorkers=6):
  outputLabel = name
  permWorkDir = os.path.abspath('swarm')
  if not os.path.exists(permWorkDir):
    os.mkdir(permWorkDir)

  modelParams = permutations_runner.runWithConfig(
    swarmConfig,
    {"maxWorkers": maxWorkers, "overwrite": True},
    outputLabel=outputLabel,
    outDir=permWorkDir,
    permWorkDir=permWorkDir,
    verbosity=0)
  modelParamsFile = writeModelParamsToFile(modelParams, name)
  return modelParamsFile


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


def runNupicModel(filePath, model, plot, useDifference=True):

  fileName = os.path.splitext(os.path.basename(filePath))[0]

  inputField = SWARM_CONFIG["includedFields"][0]['fieldName']
  predictedField = SWARM_CONFIG['inferenceArgs']['predictedField']
  predictionSteps = SWARM_CONFIG['inferenceArgs']['predictionSteps']
  nPredictionSteps = len(predictionSteps)
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

  outputFileName = './prediction/'+fileName+'_TM_pred.csv'
  outputFile = open(outputFileName,"w")
  csvWriter = csv.writer(outputFile)
  csvWriter.writerow(['step', 'data'])
  csvWriter.writerow(['int', 'float'])
  csvWriter.writerow(['', ''])

  data = pd.read_csv(filePath, header=0, skiprows=[1,2])

  predictedFieldVals = data[predictedField].astype('float')
  firstDifference = predictedFieldVals.diff()

  time_step = []
  actual_data = []
  predict_data = np.zeros((nPredictionSteps, 0))
  for i in xrange(len(data)):
    time_step.append(i)
    if (i % 100 == 0):
      print "Read %i lines..." % i

    if useDifference:
      inputRecord = {inputField: float(firstDifference.values[i])}
    else:
      inputRecord = {inputField: float(predictedFieldVals.values[i])}

    result = model.run(inputRecord)

    actual_data.append(float(predictedFieldVals.values[i]))
    prediction = result.inferences["multiStepBestPredictions"]
    prediction_values = np.array(prediction.values()).reshape((nPredictionSteps,1))
    prediction_values = np.where(prediction_values == np.array(None), 0, prediction_values)

    if useDifference:
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

    csvWriter.writerow([time_step[-1], actual_data[-1], allPrediction[0]])

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


def swarm(SWARM_CONFIG, swarmOnDifference=False):
  if swarmOnDifference:
    filePath = SWARM_CONFIG["streamDef"]['streams'][0]['source']
    name = os.path.splitext(os.path.basename(filePath))[0]
    SWARM_CONFIG["streamDef"]['streams'][0]['source'] = 'file://data/'+name+'_FirstDifference.csv'

  filePath = SWARM_CONFIG["streamDef"]['streams'][0]['source']
  filePath = filePath[7:]
  name = os.path.splitext(os.path.basename(filePath))[0]

  print "================================================="
  print "= Swarming on %s data..." % name
  print " Swam size: ", (SWARM_CONFIG["swarmSize"])
  print " Read Input File: ", filePath

  predictedField = SWARM_CONFIG['inferenceArgs']['predictedField']

  data = pd.read_csv(filePath, header=0, skiprows=[1,2])
  data = data[predictedField].astype('float')

  SWARM_CONFIG['includedFields'][0]['minValue'] = float(data.min())
  SWARM_CONFIG['includedFields'][0]['maxValue'] = float(data.max())

  pprint.pprint(SWARM_CONFIG)
  print "================================================="
  modelParams = swarmForBestModelParams(SWARM_CONFIG, name)
  print "\nWrote the following model param files:"
  print "\t%s" % modelParams

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
                    "--swarmOnDifference",
                    default=False,
                    dest="swarmOnDifference",
                    help="Set to True to use delta encoder")

  (options, remainder) = parser.parse_args()
  print options

  return options, remainder


def runExperiment(dataSet, swarmOnDifference=False):

  filePath = SWARM_CONFIG["streamDef"]['streams'][0]['source']
  filePath = filePath[7:]
  fileName = os.path.splitext(filePath)

  # calculate first difference
  if swarmOnDifference:
    filePathtrain = fileName[0] + '_FirstDifference' + fileName[1]
    calculateFirstDifference(filePath, filePathtrain)

    filePathtestOriginal = fileName[0]+'_cont'+'.csv'
    filePathtest = fileName[0] + '_FirstDifference' +'_cont'+'.csv'
    calculateFirstDifference(filePathtestOriginal, filePathtest)
  else:
    filePathtest = fileName[0] + '_cont'+'.csv'
    filePathtestOriginal = filePathtest
  swarm(SWARM_CONFIG, swarmOnDifference)

  filePath = SWARM_CONFIG["streamDef"]['streams'][0]['source']
  fileName = os.path.splitext(os.path.basename(filePath))[0]

  modelName = fileName
  modelParams = getModelParamsFromName(modelName)
  model = createModel(modelParams)

  fileTest = filePathtestOriginal
  runNupicModel(fileTest, model, plot=False, useDifference=True)


if __name__ == "__main__":

  # dataSet = 'sine'
  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  swarmOnDifference = _options.dataSet
  SWARM_CONFIG = importSwarmDescription(dataSet)
  runExperiment(dataSet, swarmOnDifference)



