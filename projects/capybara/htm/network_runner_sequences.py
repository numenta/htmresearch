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



def _newTrace():
  return {
    'label': [],
    'spActiveColumns': [],
    'tmPredictedActiveCells': [],
  }



def _updateTrace(traceNZ, label, spActiveColumns, tmPredictedActiveCells):
  traceNZ['label'].append(label)
  traceNZ['spActiveColumns'].append(spActiveColumns)
  traceNZ['tmPredictedActiveCells'].append(tmPredictedActiveCells)
  return traceNZ



def _writeTraceBatch(traceNZ, traceWriter):
  numSequences = len(traceNZ['label'])
  for i in range(numSequences):
    row = []
    for traceName in traceNZ.keys():
      row.append(json.dumps(traceNZ[traceName][i]))
    traceWriter.writerow(row)



def run(network, inputDir, inputFileName, outputDir, learningMode, chunkSize):
  # Make sure the output dir exists
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

  # Remove trace file if it already exists
  traceFileName = 'trace_%s' % inputFileName
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

      # Run the network.
      for row in reader:
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


        traceNZ = _updateTrace(traceNZ, label, spActiveColumns, 
                               tmPredictedActiveCells)

        # Write and reset trace periodically to optimize memory usage.
        sequenceNumber = int(reader.line_num)
        if sequenceNumber % chunkSize == 0:
          _writeTraceBatch(traceNZ, traceWriter)
          traceNZ = _newTrace()
          _LOGGER.info('Wrote to file (label=%s, sequenceNumber=%s)'
                       % (label, sequenceNumber))

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
                      default='configs/body_acc_x.yml',
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
  run(network, inputDir, trainFileName, outputDir, learningMode, chunkSize)

  # Run HTM on test set (learning disabled)
  learningMode = False
  _LOGGER.info('Input file: %s' % testFileName)
  run(network, inputDir, testFileName, outputDir, learningMode, chunkSize)



if __name__ == '__main__':
  main()
