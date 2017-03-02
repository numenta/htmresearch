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



SKIP_STABLE_ENCODINGS = False
MAX_STABLE_ENCODING_REPS = 1



def _processEncoding(encoding, recentEncodings):
  """
  An encoding shouldn't be processed if it is identical to all recent encodings
  """

  if len(recentEncodings) == 0:
    return True

  for e in recentEncodings:
    if not (e == encoding).all():
      return True

  return False



def _getConfig(configFilePath):
  with open(configFilePath, 'r') as ymlFile:
    config = yaml.load(ymlFile)

  inputDir = config['inputs']['inputDir']
  baseName = config['inputs']['baseName']
  metricName = config['inputs']['metricName']
  inputMin = config['inputs']['metricMin']
  inputMax = config['inputs']['metricMax']
  outputDir = config['outputs']['outputDir']
  batchSize = config['params']['batchSize']
  runSanity = config['params']['runSanity']

  return (inputDir,
          baseName,
          metricName,
          inputMin,
          inputMax,
          outputDir,
          batchSize,
          runSanity)



class NetworkRunner(object):
  def __init__(self, batchSize, inputMin, inputMax, runSanity):

    self.batchSize = batchSize
    self.network = BaseNetwork(inputMin=inputMin,
                               inputMax=inputMax,
                               runSanity=runSanity)
    self.network.initialize()
    self.traceNZ = _newTrace()

    # Keep track of the min and max value observed as we stream the data.
    self.observedMin = None
    self.observedMax = None

    # Keep track of stable encodings repetitions.
    self.recentEncodings = deque(maxlen=MAX_STABLE_ENCODING_REPS)

    # Keep track of the processing time
    self.startTime = time.time()


  def _updateTrace(self, t, scalarValue, label, encoderOutput,
                   spActiveColumns, tmActiveCells,
                   tmPredictiveCells, tmPredictedActiveCells, rawAnomalyScore):

    self.traceNZ['t'].append(t)
    self.traceNZ['scalarValue'].append(scalarValue)
    self.traceNZ['label'].append(label)
    self.traceNZ['encoderOutput'].append(encoderOutput)
    self.traceNZ['spActiveColumns'].append(spActiveColumns)
    self.traceNZ['tmActiveCells'].append(tmActiveCells)
    self.traceNZ['tmPredictiveCells'].append(tmPredictiveCells)
    self.traceNZ['tmPredictedActiveCells'].append(tmPredictedActiveCells)
    self.traceNZ['rawAnomalyScore'].append(rawAnomalyScore)


  def _writeTraceBatch(self, traceWriter):
    numPoints = len(self.traceNZ['t'])
    for i in range(numPoints):
      row = []
      for traceName in self.traceNZ.keys():
        trace = self.traceNZ[traceName]
        if trace:
          if type(trace[i]) == np.ndarray:
            trace[i] = trace[i].tolist()
          row.append(json.dumps(trace[i]))
        else:
          row.append(None)
      traceWriter.writerow(row)


  def _setObservedMinMax(self, scalarValue):
    if self.observedMin is None:
      self.observedMin = scalarValue
    elif self.observedMin > scalarValue:
      self.observedMin = scalarValue

    if self.observedMax is None:
      self.observedMax = scalarValue
    elif self.observedMax < scalarValue:
      self.observedMax = scalarValue


  def run(self, inputDir, inputFileName, inputMetricName, outputDir,
          learningMode=True):

    # Make sure the input file has the right extension
    if '.csv' not in inputFileName:
      raise ValueError('Input file needs to be in the CSV format but is %s'
                       % inputFileName)

    # Make sure the output dir exists
    if not os.path.exists(outputDir):
      os.makedirs(outputDir)

    # Remove trace file if it already exists
    traceFileName = 'trace_%s_%s' % (inputMetricName, inputFileName)
    traceFilePath = os.path.join(outputDir, traceFileName)
    if os.path.exists(traceFilePath):
      os.remove(traceFilePath)

    # Open trace file in append mode to write traces in batches.
    with open(traceFilePath, 'a') as traceFile:
      traceWriter = csv.writer(traceFile)
      traceHeaders = self.traceNZ.keys()
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
          self._setObservedMinMax(value)
          self.network.encodeValue(value)

          runNetwork = True
          # Optionally run the network if too many stable encodings are 
          # seen consecutively.
          if SKIP_STABLE_ENCODINGS:
            encoding = self.network.getEncoderOutputNZ()
            runNetwork = _processEncoding(encoding, self.recentEncodings)
            self.recentEncodings.append(encoding)

          if runNetwork:
            # The scalar value was already encoded, so don't do it again.
            self.network.handleRecord(value, label=label,
                                      skipEncoding=True,
                                      learningMode=learningMode)
            self._updateTrace(t, value, label,
                              self.network.getEncoderOutputNZ(),
                              self.network.getSpOutputNZ(),
                              self.network.getTmActiveCellsNZ(),
                              self.network.getTmPredictiveCellsNZ(),
                              self.network.getTmPredictedActiveCellsNZ(),
                              self.network.getRawAnomalyScore())

          # Write and reset trace periodically to optimize memory usage.
          if t % self.batchSize == 0:
            self._writeTraceBatch(traceWriter)
            self.traceNZ = _newTrace()
            _LOGGER.info('Wrote to file (t=%s, label=%s)' % (t, label))

            # Log elapsed time.            
            elapsedTime = time.time() - self.startTime
            _LOGGER.info('Elapsed time: %.2fs' % elapsedTime)

      # Once we are done reading, write and reset remaining traces.
      if len(self.traceNZ) > 0:
        self._writeTraceBatch(traceWriter)
        self.traceNZ = _newTrace()



def main():
  # Names of train & test phase suffix used in input CSV files naming scheme. 
  # E.g: inputFileBaseName='data' -> CSVs: 'data_train.csv' and 'data_test.csv'
  phases = ['train', 'test']

  # Parse input options.
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', '-c',
                      dest='config',
                      type=str,
                      default='config.yml',
                      help='Name of YAML config file.')
  options = parser.parse_args()
  configFile = options.config

  # Get config options.
  (inputDir,
   baseName,
   metricName,
   inputMin,
   inputMax,
   outputDir,
   batchSize,
   runSanity) = _getConfig(configFile)

  # Run the same network for each phase.
  runner = NetworkRunner(batchSize, inputMin, inputMax, runSanity)
  for phase in phases:
    inputFileName = '%s_%s.csv' % (baseName, phase)

    # Disable learning for phases other than 'train'.
    if phase == 'train':
      learningMode = True
    else:
      learningMode = False

    _LOGGER.info('Data: %s' % inputFileName)
    runner.run(inputDir, inputFileName, metricName, outputDir,
               learningMode=learningMode)

    _LOGGER.info('Traces saved in: %s/' % outputDir)
    _LOGGER.info('Observed input min=%s, max=%s' % (runner.observedMin,
                                                    runner.observedMax))



if __name__ == '__main__':
  main()
