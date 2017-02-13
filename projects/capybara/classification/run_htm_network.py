#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
import argparse
import csv
import logging
import json
import numpy as np
import simplejson
import os

from nupic.data.file_record_stream import FileRecordStream

from htmresearch.frameworks.classification.network_factory import (
  createAndConfigureNetwork, setRegionLearning, loadNetwork, saveNetwork)

_LOGGER = logging.getLogger()
_LOGGER.setLevel(logging.DEBUG)



def initTrace():
  trace = {
    'recordNumber': [],
    'sensorValue': [],
    'actualCategory': [],
    'spActiveColumns': [],
    'tmActiveCells': [],
    'tmPredictedActiveCells': [],
    'anomalyScore': [],
    'classificationInference': [],
    'classificationAccuracy': [],
  }

  return trace



def computeAccuracy(value, expectedValue):
  if value != expectedValue:
    accuracy = 0
  else:
    accuracy = 1
  return accuracy



def computeNetworkStats(sensorRegion,
                        spRegion,
                        tmRegion,
                        classifierRegion):
  """ 
  Compute HTM network statistics 
  """
  sensorValue = sensorRegion.getOutputData('sourceOut')[0]
  actualCategory = sensorRegion.getOutputData('categoryOut')[0]

  if spRegion:
    encoderNZ = spRegion.getInputData('bottomUpIn').astype(int).nonzero()[0]
    _LOGGER.debug('Encoder non-zero indices: %s' % encoderNZ)
    spActiveColumns = spRegion.getOutputData(
      'bottomUpOut').astype(int).nonzero()[0]
  else:
    spActiveColumns = None

  if tmRegion:
    tmPredictedActiveCells = tmRegion.getOutputData(
      'predictedActiveCells').astype(int).nonzero()[0]
    tmActiveCells = tmRegion.getOutputData(
      'activeCells').astype(int).nonzero()[0]
    anomalyScore = tmRegion.getOutputData(
      'anomalyScore')[0]

  else:
    tmActiveCells = None
    tmPredictedActiveCells = None
    anomalyScore = None

  classificationInference = getClassifierInference(classifierRegion)
  classificationAccuracy = computeAccuracy(classificationInference,
                                           actualCategory)

  return (sensorValue,
          actualCategory,
          spActiveColumns,
          tmActiveCells,
          tmPredictedActiveCells,
          anomalyScore,
          classificationInference,
          classificationAccuracy)



def updateTrace(trace,
                recordNumber,
                sensorValue,
                actualCategory,
                spActiveColumns,
                tmActiveCells,
                tmPredictedActiveCells,
                anomalyScore,
                classificationInference,
                classificationAccuracy):
  trace['recordNumber'].append(recordNumber)
  trace['sensorValue'].append(sensorValue)
  trace['actualCategory'].append(actualCategory)
  trace['spActiveColumns'].append(spActiveColumns)
  trace['tmActiveCells'].append(tmActiveCells)
  trace['tmPredictedActiveCells'].append(tmPredictedActiveCells)
  trace['anomalyScore'].append(anomalyScore)
  trace['classificationInference'].append(classificationInference)
  trace['classificationAccuracy'].append(classificationAccuracy)



def outputClassificationInfo(recordNumber,
                             sensorValue,
                             actualCategory,
                             anomalyScore,
                             classificationInference,
                             classificationAccuracy):
  # Network
  _LOGGER.debug('-> recordNumber: %s' % recordNumber)
  _LOGGER.debug('-> sensorValue: %s' % sensorValue)
  _LOGGER.debug('-> actualCategory: %s' % actualCategory)
  _LOGGER.debug('-> anomalyScore: %s' % anomalyScore)

  # Classification
  _LOGGER.debug('-> classificationInference: %s' % classificationInference)
  _LOGGER.debug('-> classificationAccuracy: %s / 1' % classificationAccuracy)



def getTraceFileName(filePath):
  return filePath.split('/')[-1].split('.csv')[0]



def getClassifierInference(classifierRegion):
  """Return output categories from the classifier region."""
  if classifierRegion.type == 'py.KNNClassifierRegion':
    # The use of numpy.lexsort() here is to first sort by labelFreq, then
    # sort by random values; this breaks ties in a random manner.
    inferenceValues = classifierRegion.getOutputData('categoriesOut')
    randomValues = np.random.random(inferenceValues.size)
    return np.lexsort((randomValues, inferenceValues))[-1]
  else:
    return classifierRegion.getOutputData('categoriesOut')[0]



def convertNonZeroToSDR(patternNZs, sdrSize):
  sdrs = []
  for patternNZ in patternNZs:
    sdr = np.zeros(sdrSize)
    sdr[patternNZ] = 1
    sdrs.append(sdr)

  return sdrs



def appendTraceToTraceFile(trace, traceWriter):
  numPoints = len(trace['sensorValue'])
  for i in range(numPoints):
    row = []
    for tk in trace.keys():
      if trace[tk]:
        if type(trace[tk][i]) == np.ndarray:
          row.append(json.dumps(trace[tk][i].tolist()))
        else:
          row.append(trace[tk][i])
      else:
        row.append(None)
    traceWriter.writerow(row)
  _LOGGER.info('Wrote trace batch to file.')



def createNetwork(dataSource, networkConfig, serializedModelPath):
  if serializedModelPath:
    return loadNetwork(serializedModelPath, dataSource)
  else:
    return createAndConfigureNetwork(dataSource, networkConfig)



def runNetwork(network, networkConfig, traceFilePath, numRecords, batchSize,
               learningMode):
  (sensorRegion,
   spRegion,
   tmRegion,
   _,
   classifierRegion) = setRegionLearning(network, networkConfig,
                                         learningMode=learningMode)
  trace = initTrace()

  if os.path.exists(traceFilePath):
    os.remove(traceFilePath)
  with open(traceFilePath, 'a') as traceFile:
    traceWriter = csv.writer(traceFile)
    headers = trace.keys()
    traceWriter.writerow(headers)
    for recordNumber in range(numRecords):
      network.run(1)

      (sensorValue,
       actualCategory,
       spActiveColumns,
       tmActiveCells,
       tmPredictedActiveCells,
       anomalyScore,
       classificationInference,
       classificationAccuracy) = computeNetworkStats(sensorRegion,
                                                     spRegion,
                                                     tmRegion,
                                                     classifierRegion)

      updateTrace(trace,
                  recordNumber,
                  sensorValue,
                  actualCategory,
                  spActiveColumns,
                  tmActiveCells,
                  tmPredictedActiveCells,
                  anomalyScore,
                  classificationInference,
                  classificationAccuracy)

      if recordNumber % batchSize == 0:
        outputClassificationInfo(recordNumber,
                                 sensorValue,
                                 actualCategory,
                                 anomalyScore,
                                 classificationInference,
                                 classificationAccuracy)

        # To optimize memory usage, write to trace file in batches.
        appendTraceToTraceFile(trace, traceWriter)
        trace = initTrace()

    appendTraceToTraceFile(trace, traceWriter)
  _LOGGER.info('%s records processed. Trace saved: %s' % (numRecords,
                                                          traceFilePath))



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputFile', '-d',
                      dest='inputFile',
                      type=str,
                      default=None,
                      help='Relative path to the input file.')

  parser.add_argument('--outputDir', '-o',
                      dest='outputDir',
                      type=str,
                      default='results/traces',
                      help='Relative path to the directory where the HTM '
                           'network traces will be saved.')

  parser.add_argument('--htmConfig', '-c',
                      dest='htmConfig',
                      type=str,
                      default='htm_network_config/6categories.json',
                      help='Relative path to the HTM network config JSON. '
                           'This option is ignored when the --model flag '
                           'is used.')

  parser.add_argument('--inputModel', '-im',
                      dest='inputModel',
                      type=str,
                      default=None,
                      help='Relative path of the serialized HTM model to be '
                           'loaded.')

  parser.add_argument('--outputModel', '-om',
                      dest='outputModel',
                      type=str,
                      default=None,
                      help='Relative path to serialize the HTM model.')

  parser.add_argument('--disableLearning', '-dl',
                      dest='disableLearning',
                      action='store_true',
                      default=False,
                      help='Use this flag to disable learning. If not '
                           'provided, then learning is enabled by default.')

  parser.add_argument('--batch', '-b',
                      dest='batchSize',
                      type=int,
                      default=1000,
                      help='Size of each batch being processed.')

  # Parse input options
  options = parser.parse_args()
  outputDir = options.outputDir
  networkConfigPath = options.htmConfig
  batchSize = options.batchSize

  # FIXME RES-464: until the serialization process is fixed, don't save the 
  # model .
  # Run serially each phase (train -> validation -> test) and override:
  # - inputFile
  # - inputModelPath
  # - outputModelPath
  # - learningMode

  # TODO: Re-introduce these command line args when serialization is fixed.
  # inputFile = options.inputFile
  # inputModelPath = options.inputModel
  # outputModelPath = options.outputModel
  # learningMode = not options.disableLearning

  inputModelPath = None
  outputModelPath = None
  phases = ['train', 'val', 'test']
  inputDir = os.path.join('data', 'artificial')
  expName = 'binary_ampl=10.0_mean=0.0_noise=0.0'  # 'body_acc_x_inertial_signals'  
  network = None
  with open(networkConfigPath, 'r') as f:
    networkConfig = simplejson.load(f)
    for phase in phases:

      # Data source
      inputFile = os.path.join(inputDir, '%s_%s.csv' % (expName, phase))
      dataSource = FileRecordStream(streamID=inputFile)
      numRecords = dataSource.getDataRowCount()
      _LOGGER.debug('Number of records to be processed: %s' % numRecords)

      # Trace output info
      traceFileName = getTraceFileName(inputFile)
      traceFilePath = os.path.join(outputDir, '%s.csv' % traceFileName)
      if not os.path.exists(outputDir):
        os.makedirs(outputDir)

      # If there is not network, create one and train it.
      if not network:
        assert phase == 'train'  # Make sure that we create a network for 
        learningMode = True
        network = createNetwork(dataSource, networkConfig, inputModelPath)
      else:
        learningMode = False
        regionName = networkConfig["sensorRegionConfig"]["regionName"]
        sensorRegion = network.regions[regionName].getSelf()
        sensorRegion.dataSource = dataSource
        if 'train' in sensorRegion.dataSource._filename:
          raise ValueError('Learning mode should not be disabled for the '
                           'train set.')

      _LOGGER.debug('Running network with inputFile=%s '
                    'and learningMode=%s' % (inputFile, learningMode))

      # FIXME RES-464 (end)

      run(network,
          numRecords,
          traceFilePath,
          networkConfig,
          outputModelPath,
          batchSize,
          learningMode)



def run(network,
        numRecords,
        traceFilePath,
        networkConfig,
        outputModelPath,
        batchSize,
        learningMode):
  # Check input options
  if outputModelPath and os.path.exists(outputModelPath):
    _LOGGER.warning('There is already a model named %s. This model will be '
                    'erased.' % outputModelPath)

  # HTM network
  runNetwork(network, networkConfig, traceFilePath, numRecords, batchSize,
             learningMode)

  # Save network
  if outputModelPath:
    saveNetwork(network, outputModelPath)



if __name__ == '__main__':
  main()
