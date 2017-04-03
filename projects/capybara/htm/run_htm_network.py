#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
import json
import logging
import os
import time
import yaml
import numpy as np

from htm_network import BaseNetwork

logging.basicConfig()
_LOGGER = logging.getLogger('NetworkRunner')
_LOGGER.setLevel(logging.INFO)



def _getConfig(configFilePath):
  with open(configFilePath, 'r') as ymlFile:
    config = yaml.load(ymlFile)

  inputDir = config['inputs']['inputDir']
  trainFileName = config['inputs']['trainFileName']
  testFileName = config['inputs']['testFileName']
  timeIndexed = config['inputs']['timeIndexed']
  metricName = config['inputs']['metricName']
  inputMin = config['inputs']['metricMin']
  inputMax = config['inputs']['metricMax']
  outputDir = config['outputs']['outputDir']
  chunkSize = config['params']['chunkSize']
  runSanity = config['params']['runSanity']

  return (inputDir,
          trainFileName,
          testFileName,
          timeIndexed,
          metricName,
          inputMin,
          inputMax,
          outputDir,
          chunkSize,
          runSanity)



def _newTrace(timeIndexed):
  traceNZ = {
    'label': [],
    'spActiveColumns': [],
    'tmPredictedActiveCells': []
  }

  if timeIndexed:
    traceNZ['tmActiveCells'] = []
    traceNZ['tmPredictiveCells'] = []
    traceNZ['encoderOutput'] = []
    traceNZ['rawAnomalyScore'] = []
    traceNZ['scalarValue'] = []
    traceNZ['t'] = []

  return traceNZ



def _updateTrace(traceNZ, **kwargs):
  for k, v in kwargs.items():
    traceNZ[k].append(v)
  return traceNZ



def _writeTraceBatch(traceNZ, traceWriter, timeIndexed):
  if timeIndexed:
    _writeTimeIndexedTrace(traceNZ, traceWriter)
  else:
    _writeSequenceIndexedTrace(traceNZ, traceWriter)



def _writeTimeIndexedTrace(traceNZ, traceWriter):
  numPoints = len(traceNZ['t'])
  for i in range(numPoints):
    row = []
    for traceName in traceNZ.keys():
      trace = traceNZ[traceName]
      if trace:
        if type(trace[i]) == np.ndarray:
          trace[i] = trace[i].tolist()
        row.append(json.dumps(trace[i]))
      else:
        row.append(None)
    traceWriter.writerow(row)



def _writeSequenceIndexedTrace(traceNZ, traceWriter):
  numSequences = len(traceNZ['label'])
  for i in range(numSequences):
    row = []
    for traceName in traceNZ.keys():
      row.append(json.dumps(traceNZ[traceName][i]))
    traceWriter.writerow(row)



def _runOnTimeIndexedData(network, learningMode, traceCsvWriter, writeChunkSize,
                          inputCsvReader, inputMetricName):
  timeIndexed = True
  startTime = time.time()

  traceNZ = _newTrace(timeIndexed)
  traceHeaders = traceNZ.keys()
  traceCsvWriter.writerow(traceHeaders)

  inputHeaders = inputCsvReader.next()
  for row in inputCsvReader:
    t = int(inputCsvReader.line_num)
    data = dict(zip(inputHeaders, row))
    label = json.loads(data['label'])
    value = json.loads(data[inputMetricName])
    network.handleRecord(value, label=label,
                         learningMode=learningMode)

    traceUpdate = {
      'label': label,
      'spActiveColumns': network.getSpOutputNZ(),
      'tmPredictedActiveCells': network.getTmPredictedActiveCellsNZ(),
      'tmActiveCells': network.getTmActiveCellsNZ(),
      'tmPredictiveCells': network.getTmPredictiveCellsNZ(),
      'encoderOutput': network.getEncoderOutputNZ(),
      'rawAnomalyScore': network.getRawAnomalyScore(),
      'scalarValue': value, 't': t
    }

    traceNZ = _updateTrace(traceNZ, **traceUpdate)

    # Write and reset trace periodically to optimize memory usage.
    if t % writeChunkSize == 0:
      _writeTraceBatch(traceNZ, traceCsvWriter, timeIndexed)
      traceNZ = _newTrace(timeIndexed)
      _LOGGER.info('Wrote to file (t=%s, label=%s)' % (t, label))

      elapsedTime = time.time() - startTime
      _LOGGER.info('Elapsed time: %.2fs' % elapsedTime)

  # Once we are done reading, write and reset remaining traces.
  if len(traceNZ) > 0:
    _writeTraceBatch(traceNZ, traceCsvWriter, timeIndexed)



def _runOnSequenceIndexedData(network, learningMode, traceCsvWriter,
                              writeChunkSize, inputCsvReader):
  timeIndexed = False
  startTime = time.time()

  traceNZ = _newTrace(timeIndexed)
  traceHeaders = traceNZ.keys()
  traceCsvWriter.writerow(traceHeaders)

  for row in inputCsvReader:
    label = int(float(row[0]))
    sequence_values = row[1:]

    spActiveColumns = []
    tmPredictedActiveCells = []
    for valueString in sequence_values:
      value = float(valueString)
      network.handleRecord(value, label=label, learningMode=learningMode)
      spActiveColumns.append(network.getSpOutputNZ().tolist())
      tmPredictedActiveCells.append(
        network.getTmPredictedActiveCellsNZ().tolist())

    traceUpdate = {
      'label': label, 'spActiveColumns': spActiveColumns,
      'tmPredictedActiveCells': tmPredictedActiveCells
    }
    traceNZ = _updateTrace(traceNZ, **traceUpdate)

    # Write and reset trace periodically to optimize memory usage.
    sequenceNumber = int(inputCsvReader.line_num)
    if sequenceNumber % writeChunkSize == 0:
      _writeTraceBatch(traceNZ, traceCsvWriter, timeIndexed)
      traceNZ = _newTrace(timeIndexed)
      _LOGGER.info('Wrote to file (label=%s, sequenceNumber=%s)'
                   % (label, sequenceNumber))

      # Log elapsed time.
      elapsedTime = time.time() - startTime
      _LOGGER.info('Elapsed time: %.2fs' % elapsedTime)

  # Once we are done reading, write and reset remaining traces.
  if len(traceNZ) > 0:
    _writeTraceBatch(traceNZ, traceCsvWriter, timeIndexed)



def run(network, inputDir, inputFileName, inputMetricName, outputDir,
        learningMode, chunkSize, timeIndexed):
  # Make sure the output dir exists
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

  # Remove trace file if it already exists
  traceFileName = 'trace_%s_%s' % (inputMetricName, inputFileName)
  traceFilePath = os.path.join(outputDir, traceFileName)
  if os.path.exists(traceFilePath):
    os.remove(traceFilePath)

  # Open trace file in append mode to write traces in chunks.
  with open(traceFilePath, 'a') as traceFile:
    traceCsvWriter = csv.writer(traceFile)

    # Run network on input data.
    inputFilePath = os.path.join(inputDir, inputFileName)
    with open(inputFilePath, 'r') as inputFile:
      inputCsvReader = csv.reader(inputFile)
      if timeIndexed:
        _runOnTimeIndexedData(network, learningMode, traceCsvWriter, chunkSize,
                              inputCsvReader, inputMetricName)
      else:
        _runOnSequenceIndexedData(network, learningMode, traceCsvWriter,
                                  chunkSize, inputCsvReader)
      _LOGGER.info('Traces saved in: %s/' % outputDir)



def main():
  # Parse input options.
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', '-c',
                      dest='config',
                      type=str,
                      default='configs/body_acc_x.yml',
                      help='Name of YML config file.')
  options = parser.parse_args()
  configFile = options.config

  (inputDir,
   trainFileName,
   testFileName,
   timeIndexed,
   metricName,
   inputMin,
   inputMax,
   outputDir,
   chunkSize,
   runSanity) = _getConfig(configFile)

  # Initialize the network.
  network = BaseNetwork(inputMin=inputMin, inputMax=inputMax,
                        runSanity=runSanity)
  network.initialize()

  # Run HTM on train set
  learningMode = True
  _LOGGER.info('Input file: %s' % trainFileName)
  run(network, inputDir, trainFileName, metricName, outputDir,
      learningMode, chunkSize, timeIndexed)

  # Run HTM on test set (learning disabled)
  learningMode = False
  _LOGGER.info('Input file: %s' % testFileName)
  run(network, inputDir, testFileName, metricName, outputDir,
      learningMode, chunkSize, timeIndexed)



if __name__ == '__main__':
  main()
