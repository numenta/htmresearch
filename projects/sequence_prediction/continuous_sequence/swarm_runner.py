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

import os, csv
import pprint
import importlib

from optparse import OptionParser
from nupic.swarming import permutations_runner
import pandas as pd
from nupic.swarming.exp_generator.ExpGenerator import expGenerator
import json

class SwarmRunner():
  def __init__(self, dataSet, useDeltaEncoder=False):
    self.dataSet = dataSet
    self.useDeltaEncoder = useDeltaEncoder

  @staticmethod
  def modelParamsToString(modelParams):
    pp = pprint.PrettyPrinter(indent=2)
    return pp.pformat(modelParams)

  @staticmethod
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

  @staticmethod
  def writeModelParamsToFile(modelParams, name):
    cleanName = name.replace(" ", "_").replace("-", "_")
    paramsName = "%s_model_params.py" % cleanName
    outDir = os.path.join(os.getcwd(), 'model_params')
    if not os.path.isdir(outDir):
      os.mkdir(outDir)
    outPath = os.path.join(os.getcwd(), 'model_params', paramsName)
    with open(outPath, "wb") as outFile:
      modelParamsString = SwarmRunner.modelParamsToString(modelParams)
      outFile.write("MODEL_PARAMS = \\\n%s" % modelParamsString)
    return outPath

  @staticmethod
  def importSwarmDescription(dataSet):
    swarmConfigFileName = 'SWARM_CONFIG_' + dataSet
    try:
      SWARM_CONFIG = importlib.import_module("swarm_description.%s" % swarmConfigFileName).SWARM_CONFIG
    except ImportError:
      raise Exception("No swarm_description exist for '%s'. Create swarm_description first"
                      % dataSet)
    return SWARM_CONFIG

  def swarmForBestModelParams(self, swarmConfig, name, maxWorkers=6):
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
    modelParamsFile = self.writeModelParamsToFile(modelParams, name)
    return modelParamsFile

  def generateExperimentDescription(self, SWARM_CONFIG, outDir='./swarm/'):
    expDescConfig = json.dumps(SWARM_CONFIG)
    expDescConfig = expDescConfig.splitlines()
    expDescConfig = "".join(expDescConfig)

    expGenerator([
      "--description=%s" % (expDescConfig),
      "--outDir=%s" % (outDir)])


  def getSourceFile(self, SWARM_CONFIG):
    filePath = SWARM_CONFIG["streamDef"]['streams'][0]['source']
    filePath = filePath[7:]
    fileName = os.path.splitext(filePath)[0]

    # calculate first difference if delta encoder is used
    if self.useDeltaEncoder:
      filePathtrain = fileName + '_FirstDifference' + '.csv'
      self.calculateFirstDifference(filePath, filePathtrain)
    else:
      filePathtrain = fileName + '.csv'

    return filePathtrain

  def runExperiment(self, SWARM_CONFIG):

    filePathtrain = self.getSourceFile(SWARM_CONFIG)

    # run swarm on data file
    SWARM_CONFIG["streamDef"]['streams'][0]['source'] = 'file://'+filePathtrain
    name = os.path.splitext(os.path.basename(filePathtrain))[0]

    print "================================================="
    print "= Swarming on %s data..." % name
    print " Swam size: ", (SWARM_CONFIG["swarmSize"])
    print " Read Input File: ", filePathtrain

    # adjust min/max value based on data
    # predictedField = SWARM_CONFIG['inferenceArgs']['predictedField']
    # data = pd.read_csv(filePath, header=0, skiprows=[1,2])
    # data = data[predictedField].astype('float')
    # SWARM_CONFIG['includedFields'][0]['minValue'] = float(data.min())
    # SWARM_CONFIG['includedFields'][0]['maxValue'] = float(data.max())

    pprint.pprint(SWARM_CONFIG)
    print "================================================="
    modelParams = self.swarmForBestModelParams(SWARM_CONFIG, name)
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
                    "--useDeltaEncoder",
                    default=False,
                    dest="useDeltaEncoder",
                    help="Set to True to use delta encoder")

  parser.add_option("-g",
                    "--generateDescriptionFile",
                    default=False,
                    dest="generateDescriptionFile",
                    help="Set to True to generate swarm description file")

  (options, remainder) = parser.parse_args()
  print options

  return options, remainder


if __name__ == "__main__":
  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  useDeltaEncoder = _options.useDeltaEncoder

  swarm_runner = SwarmRunner(dataSet, useDeltaEncoder)
  SWARM_CONFIG = SwarmRunner.importSwarmDescription(dataSet)

  if _options.generateDescriptionFile:
    # only generate description file for swarming
    filePathTrain = swarm_runner.getSourceFile(SWARM_CONFIG)
    swarm_runner.generateExperimentDescription(SWARM_CONFIG, outDir='./swarm/'+dataSet)
    print " generate swarm description file for ", filePathTrain
  else:
    swarm_runner.runExperiment(SWARM_CONFIG)