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

import csv
import json
import simplejson
import numpy as np

from nupic.data.file_record_stream import FileRecordStream
from prettytable import PrettyTable

from htmresearch.frameworks.classification.network_factory import (
  configureNetwork)
from htmresearch.frameworks.classification.utils.sensor_data import (
  generateSensorData, plotSensorData, cleanTitle)
from htmresearch.frameworks.classification.utils.network_config import (
  generateSampleNetworkConfig)
from htmresearch.frameworks.classification.utils.traces import plotTraces

from settings import (NUM_CATEGORIES,
                      NUM_PHASES,
                      NUM_REPS,
                      SIGNAL_TYPES,
                      WHITE_NOISE_AMPLITUDES,
                      SIGNAL_AMPLITUDES,
                      SIGNAL_MEANS,
                      DATA_DIR,
                      USE_CONFIG_TEMPLATE,
                      NOISE_LENGTHS,
                      PLOT)



def initTrace():
  trace = {
    'step': [],
    'tmActiveCells': [],
    'tmPredictedActiveCells': [],
    'tpActiveCells': [],
    'sensorValue': [],
    'actualCategory': [],
    'predictedCategory': [],
    'classificationAccuracy': []
  }
  return trace



def rollingAccuracy(trace, expSetup, ignoreNoise=True):
  rollingWindowSize = expSetup['sequenceLength']

  now = len(trace['sensorValue'])
  start = max(0, now - rollingWindowSize)
  end = now
  window = range(start, end)

  numRecords = 0
  numNoisyRecords = 0
  correctlyClassified = 0
  for i in window:
    if ignoreNoise:
      if trace['actualCategory'][-1] > 0:
        if trace['predictedCategory'][i] == trace['actualCategory'][i]:
          correctlyClassified += 1
        numRecords += 1
      else:
        numNoisyRecords += 1
        k = start - numNoisyRecords
        if trace['predictedCategory'][k] == trace['actualCategory'][k]:
          correctlyClassified += 1
        numRecords += 1
    else:
      if trace['predictedCategory'][i] == trace['actualCategory'][i]:
        correctlyClassified += 1
      numRecords += 1

  accuracy = round(100.0 * correctlyClassified / numRecords, 2)
  return accuracy



def onlineRollingAccuracy(trace, expSetup, ignoreNoise=True):
  """
  From: http://www.daycounter.com/LabBook/Moving-Average.phtml
  :param trace: 
  :param expSetup: 
  :param ignoreNoise: 
  :return: 
  """
  #  initialize MA
  if len(trace['classificationAccuracy']) > 0:
    ma = trace['classificationAccuracy'][-1]
  else:
    ma = 0

  if trace['predictedCategory'][-1] == trace['actualCategory'][-1]:
    x = 1
  else:
    x = 0

  rollingWindowSize = expSetup['sequenceLength']
  if ignoreNoise and trace['actualCategory'][-1] > 0:
    ma += float(x - ma) / rollingWindowSize

  return ma



def updateTrace(trace, expSetup, recordNumber, sensorRegion, tmRegion, tpRegion,
                classifierRegion):
  trace['step'].append(recordNumber)

  if tpRegion:
    tpActiveCells = tpRegion.getOutputData("mostActiveCells")
    tpActiveCells = tpActiveCells.nonzero()[0]
    trace['tpActiveCells'].append(tpActiveCells)

  if tmRegion:
    tmPredictedActiveCells = tmRegion.getOutputData("predictedActiveCells")
    tmPredictedActiveCells = tmPredictedActiveCells.nonzero()[0]
    tmActiveCells = tmRegion.getOutputData("activeCells")
    tmActiveCells = tmActiveCells.nonzero()[0]
    trace['tmActiveCells'].append(tmActiveCells)
    trace['tmPredictedActiveCells'].append(tmPredictedActiveCells)

  trace['sensorValue'].append(sensorRegion.getOutputData("sourceOut")[0])

  predictedCategory = getClassifierInference(classifierRegion)
  trace['predictedCategory'].append(predictedCategory)

  actualCategory = sensorRegion.getOutputData("categoryOut")[0]
  trace['actualCategory'].append(actualCategory)

  # accuracy = rollingAccuracy(trace, expSetup)
  accuracy = onlineRollingAccuracy(trace, expSetup)

  trace['classificationAccuracy'].append(accuracy)
  return trace



def getClassifierInference(classifierRegion):
  """Return output categories from the classifier region."""
  if classifierRegion.type == "py.KNNClassifierRegion":
    # The use of numpy.lexsort() here is to first sort by labelFreq, then
    # sort by random values; this breaks ties in a random manner.
    inferenceValues = classifierRegion.getOutputData("categoriesOut")
    randomValues = np.random.random(inferenceValues.size)
    return np.lexsort((randomValues, inferenceValues))[-1]
  else:
    return classifierRegion.getOutputData("categoriesOut")[0]



def updateExpSetup(expSetup, networkConfig):
  assert isinstance(expSetup, dict)

  spEnabled = networkConfig["sensorRegionConfig"].get(
    "regionEnabled")
  tmEnabled = networkConfig["tmRegionConfig"].get(
    "regionEnabled")
  upEnabled = networkConfig["tpRegionConfig"].get(
    "regionEnabled")
  classifierType = networkConfig[
    "classifierRegionConfig"].get(
    "regionType")

  cells = networkConfig["tmRegionConfig"]["regionParams"]["cellsPerColumn"]
  columns = networkConfig["tmRegionConfig"]["regionParams"]["columnCount"]

  expSetup['spEnabled'] = spEnabled
  expSetup['tmEnabled'] = tmEnabled
  expSetup['upEnabled'] = upEnabled
  expSetup['classifierType'] = classifierType
  expSetup['numTmCells'] = cells * columns
  return expSetup



def generateExpId(expSetup, baseId):
  return '%s_sp-%s_tm-%s_tp-%s_%s' % (baseId,
                                      expSetup['spEnabled'],
                                      expSetup['tmEnabled'],
                                      expSetup['upEnabled'],
                                      expSetup['classifierType'][3:-6])



def enableRegionsLearning(network, networkConfig):
  sensorRegion = network.regions[
    networkConfig["sensorRegionConfig"].get("regionName")]

  if networkConfig["spRegionConfig"].get("regionEnabled"):
    spRegion = network.regions[
      networkConfig["spRegionConfig"].get("regionName")]
    spRegion.setParameter("learningMode", True)
  else:
    spRegion = None

  if networkConfig["tmRegionConfig"].get("regionEnabled"):
    tmRegion = network.regions[
      networkConfig["tmRegionConfig"].get("regionName")]
    tmRegion.setParameter("learningMode", True)
  else:
    tmRegion = None

  if networkConfig['tpRegionConfig'].get('regionEnabled'):
    tpRegion = network.regions[
      networkConfig['tpRegionConfig'].get('regionName')]
    tpRegion.setParameter("learningMode", True)
  else:
    tpRegion = None

  classifierRegion = network.regions[
    networkConfig["classifierRegionConfig"].get("regionName")]
  classifierRegion.setParameter("learningMode", True)

  return sensorRegion, spRegion, tmRegion, tpRegion, classifierRegion



def runNetwork(networkConfig,
               signalType,
               inputDataDir,
               numPhases,
               numReps,
               signalMean,
               signalAmplitude,
               numCategories,
               noiseAmplitude,
               noiseLengths):
  expSetup = generateSensorData(signalType,
                                inputDataDir,
                                numPhases,
                                numReps,
                                signalMean,
                                signalAmplitude,
                                numCategories,
                                noiseAmplitude,
                                noiseLengths)

  expSetup = updateExpSetup(expSetup, networkConfig)

  expId = generateExpId(expSetup, signalType)

  dataSource = FileRecordStream(streamID=expSetup['inputFilePath'])
  network = configureNetwork(dataSource, networkConfig)

  (sensorRegion,
   spRegion,
   tmRegion,
   tpRegion,
   classifierRegion) = enableRegionsLearning(network, networkConfig)

  trace = initTrace()
  for i in range(expSetup['numPoints']):
    network.run(1)
    trace = updateTrace(trace,
                        expSetup,
                        i,
                        sensorRegion,
                        tmRegion,
                        tpRegion,
                        classifierRegion)
    if i % 50 == 0:
      print '=> Record: %s | Final accuracy: %s' % (
        i, trace['classificationAccuracy'][-1])

  return {'expId': expId, 'expTrace': trace, 'expSetup': expSetup}



def runExperiments():
  if USE_CONFIG_TEMPLATE:
    with open("config/network_config_template.json", "rb") as jsonFile:
      templateNetworkConfig = simplejson.load(jsonFile)
      networkConfigurations = generateSampleNetworkConfig(templateNetworkConfig,
                                                          NUM_CATEGORIES)
  else:
    with open('config/knn_network_configs.json', 'rb') as jsonFile:
      networkConfigurations = simplejson.load(jsonFile)

  expResults = []
  for signalType in SIGNAL_TYPES:
    for networkConfig in networkConfigurations:
      for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
        for signalMean in SIGNAL_MEANS:
          for signalAmplitude in SIGNAL_AMPLITUDES:
            for numCategories in NUM_CATEGORIES:
              for numReps in NUM_REPS:
                for numPhases in NUM_PHASES:
                  for noiseLengths in NOISE_LENGTHS:
                    print 'Exp #%s' % len(expResults)
                    expResult = runNetwork(networkConfig,
                                           signalType,
                                           DATA_DIR,
                                           numPhases,
                                           numReps,
                                           signalMean,
                                           signalAmplitude,
                                           numCategories,
                                           noiseAmplitude,
                                           noiseLengths)
                    expResults.append(expResult)

  return expResults



def printExpSetups(headers, rows):
  t = PrettyTable(headers)
  for row in rows:
    t.add_row(row)
  print '%s\n' % t



def saveExpSetups(outFile, expResults):
  """
  Save exp setups and final accuracy result to CSV file
  :param outFile: (str) path to CSV file where to save data
  :param expResults: (list of dict) experiment results
  """

  rows = []
  headers = expResults[0]['expSetup'].keys()
  headers.append('finalClassificationAccuracy')

  with open(outFile, 'wb') as fw:
    writer = csv.writer(fw)
    writer.writerow(headers)
    for i in range(len(expResults)):
      row = [expResults[i]['expSetup'][h] for h in headers[:-1]]
      row.append(expResults[i]['expTrace']['classificationAccuracy'][-1])
      writer.writerow(row)
      rows.append(row)

    print '==> Results saved to %s\n' % outFile
    return headers, rows



def saveTraces(baseOutFile, expResults):
  """
  Save experiments network traces to CSV
  :param baseOutFile: (str) base name of the output file.
  :param expResults: (list of dict) experiment results
  """
  for expResult in expResults:
    expSetup = expResult['expSetup']
    expTrace = expResult['expTrace']
    outFile = baseOutFile % expResult['expId']
    with open(outFile, 'wb') as f:
      writer = csv.writer(f)
      headers = expTrace.keys()
      writer.writerow(headers)
      for i in range(expSetup['numPoints']):
        row = []
        for t in expTrace.keys():
          if len(expTrace[t]) > i:
            if type(expTrace[t][i]) == np.ndarray:
              expTrace[t][i] = list(expTrace[t][i])
            if type(expTrace[t][i]) != list:
              row.append(expTrace[t][i])
            else:
              row.append(json.dumps(expTrace[t][i]))
          else:
            row.append(None)
        writer.writerow(row)



def plotExpTraces(expResults):
  for expResult in expResults:
    xlim = [0, expResult['expSetup']['numPoints']]
    numTmCells = expResult['expSetup']['numTmCells']
    traces = expResult['expTrace']
    title = cleanTitle(expResult)
    plotTraces(numTmCells, title, xlim, traces)



def main():
  EXP_SETUPS_OUTFILE = 'results/seq_classification_results.csv'
  EXP_TRACES_OUTFILE = 'results/traces_%s.csv'

  expResults = runExperiments()

  headers, rows = saveExpSetups(EXP_SETUPS_OUTFILE, expResults)
  printExpSetups(headers, rows)

  saveTraces(EXP_TRACES_OUTFILE, expResults)

  if PLOT:
    # plotSensorData(expResults)
    plotExpTraces(expResults)



if __name__ == '__main__':
  main()
