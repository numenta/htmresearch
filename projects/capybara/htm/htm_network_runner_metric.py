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
import numpy as np
import os
import time
import yaml
from collections import deque

from htm_network import BaseNetwork

logging.basicConfig()
_LOGGER = logging.getLogger('NetworkRunner')
_LOGGER.setLevel(logging.INFO)



def _newTrace():
  return {
    't': [],
    'scalarValue': [],
    'encoderOutput': [],
    'spActiveColumns': [],
    'tmActiveCells': [],
    'tmPredictiveCells': [],
    'tmPredictedActiveCells': [],
    'rawAnomalyScore': [],
    'label': []
  }



def _getConfig(configFilePath):
  with open(configFilePath, 'r') as ymlFile:
    config = yaml.load(ymlFile)

  inputDir = config['inputs']['inputDir']
  trainFileName = config['inputs']['trainFileName']
  testFileName = config['inputs']['testFileName']
  metricName = config['inputs']['metricName']
  inputMin = config['inputs']['metricMin']
  inputMax = config['inputs']['metricMax']
  outputDir = config['outputs']['outputDir']
  chunkSize = config['params']['chunkSize']
  runSanity = config['params']['runSanity']

  return (inputDir,
          trainFileName,
          testFileName,
          metricName,
          inputMin,
          inputMax,
          outputDir,
          chunkSize,
          runSanity)



def _updateTrace(traceNZ, t, scalarValue, label, encoderOutput,
                 spActiveColumns, tmActiveCells,
                 tmPredictiveCells, tmPredictedActiveCells, rawAnomalyScore):
  traceNZ['t'].append(t)
  traceNZ['scalarValue'].append(scalarValue)
  traceNZ['label'].append(label)
  traceNZ['encoderOutput'].append(encoderOutput)
  traceNZ['spActiveColumns'].append(spActiveColumns)
  traceNZ['tmActiveCells'].append(tmActiveCells)
  traceNZ['tmPredictiveCells'].append(tmPredictiveCells)
  traceNZ['tmPredictedActiveCells'].append(tmPredictedActiveCells)
  traceNZ['rawAnomalyScore'].append(rawAnomalyScore)
  return traceNZ



def _writeTraceBatch(traceNZ, traceWriter):
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



def run(network, inputDir, inputFileName, inputMetricName, outputDir,
        learningMode, chunkSize):
  # Make sure the output dir exists
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

  # Remove trace file if it already exists
  traceFileName = 'trace_%s_%s' % (inputMetricName, inputFileName)
  traceFilePath = os.path.join(outputDir, traceFileName)
  if os.path.exists(traceFilePath):
    os.remove(traceFilePath)

  # Open trace file in append mode to write traces in chunks.
  traceNZ = _newTrace()
  startTime = time.time()
  with open(traceFilePath, 'a') as traceFile:
    traceWriter = csv.writer(traceFile)
    traceHeaders = traceNZ.keys()
    traceWriter.writerow(traceHeaders)

    # Read input file.      
    inputFilePath = os.path.join(inputDir, inputFileName)
    with open(inputFilePath, 'r') as inputFile:
      reader = csv.reader(inputFile)
      inputHeaders = reader.next()

      # Run the network.
      for row in reader:
        t = int(reader.line_num)
        data = dict(zip(inputHeaders, row))
        label = json.loads(data['label'])
        value = json.loads(data[inputMetricName])

        network.handleRecord(value, label=label,
                             learningMode=learningMode)

        traceNZ = _updateTrace(traceNZ, t, value, label,
                               network.getEncoderOutputNZ(),
                               network.getSpOutputNZ(),
                               network.getTmActiveCellsNZ(),
                               network.getTmPredictiveCellsNZ(),
                               network.getTmPredictedActiveCellsNZ(),
                               network.getRawAnomalyScore())

        # Write and reset trace periodically to optimize memory usage.
        if t % chunkSize == 0:
          _writeTraceBatch(traceNZ, traceWriter)
          traceNZ = _newTrace()
          _LOGGER.info('Wrote to file (t=%s, label=%s)' % (t, label))

          # Log elapsed time.            
          elapsedTime = time.time() - startTime
          _LOGGER.info('Elapsed time: %.2fs' % elapsedTime)

    # Once we are done reading, write and reset remaining traces.
    if len(traceNZ) > 0:
      _writeTraceBatch(traceNZ, traceWriter)
    _LOGGER.info('Traces saved in: %s/' % outputDir)



def main():
  # Parse input options.
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', '-c',
                      dest='config',
                      type=str,
                      default='configs/body_acc_x.metric.yml',
                      help='Name of YML config file.')
  options = parser.parse_args()
  configFile = options.config

  (inputDir,
   trainFileName,
   testFileName,
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
      learningMode, chunkSize)

  # Run HTM on test set (learning disabled)
  learningMode = False
  _LOGGER.info('Input file: %s' % testFileName)
  run(network, inputDir, testFileName, metricName, outputDir,
      learningMode, chunkSize)



if __name__ == '__main__':
  main()
