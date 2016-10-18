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
import logging
import numpy as np
import simplejson
import os

from htmresearch.frameworks.clustering.sdr_clustering import Clustering
from nupic.data.file_record_stream import FileRecordStream

from htmresearch.frameworks.clustering.distances import interClusterDistances
from htmresearch.frameworks.classification.network_factory import (
  configureNetwork, enableRegionLearning)
from htmresearch.frameworks.classification.utils.traces import (loadTraces,
                                                                plotTraces)
from htmresearch.frameworks.clustering.viz import (
  vizInterSequenceClusters, vizInterCategoryClusters)

from settings.htm_network import (OUTPUT_DIR,
                                  INPUT_FILES,
                                  PLOT_RESULTS,
                                  HTM_NETWORK_CONFIGS,
                                  CLUSTERING,
                                  RESULTS_OUTPUT_FILE,
                                  TRACES_OUTPUT_FILE,
                                  FILE_NAMES,
                                  MERGE_THRESHOLD,
                                  ANOMALOUS_THRESHOLD,
                                  STABLE_THRESHOLD,
                                  MIN_CLUSTER_SIZE,
                                  SIMILARITY_THRESHOLD,
                                  ROLLING_ACCURACY_WINDOW,
                                  CELLS_TO_CLUSTER,
                                  IGNORE_NOISE,
                                  ANOMALY_SCORE)

_LOGGER = logging.getLogger()
_LOGGER.setLevel(logging.DEBUG)



def initTrace(runClustering):
  trace = {
    'recordNumber': [],
    'sensorValue': [],
    'actualCategory': [],
    'tmActiveCells': [],
    'tmPredictedActiveCells': [],
    'rawAnomalyScore': [],
    'rollingAnomalyScore': [],
    'tpActiveCells': [],
    'classificationInference': [],
    'rawClassificationAccuracy': [],
    'rollingClassificationAccuracy': []
  }
  if runClustering:
    trace['clusteringInference'] = []
    trace['predictedClusterId'] = []
    trace['rollingClusteringAccuracy'] = []
    trace['clusterHomogeneity'] = []
    trace['clusteringConfidence'] = []
    trace['rawClusteringAccuracy'] = []

  return trace



def computeAccuracy(value, expectedValue):
  if value is None:
    return None

  if value != expectedValue:
    accuracy = 0
  else:
    accuracy = 1
  return accuracy



def movingAverage(trace,
                  maTraceFieldName,
                  rollingWindowSize,
                  newValue,
                  actualCategory,
                  ignoreNoise=False):
  """
  Online computation of moving average.
  From: http://www.daycounter.com/LabBook/Moving-Average.phtml
  """
  if len(trace[maTraceFieldName]) > 0:
    ma = trace[maTraceFieldName][-1]
    if newValue is None:
      return ma
    else:
      if ignoreNoise:
        if actualCategory != 0:  # noise is labelled as 0
          x = newValue
        else:
          x = ma
      else:
        x = newValue
      ma += float(x - ma) / rollingWindowSize

  else:
    ma = 0

  return ma



def updateClusteringTrace(trace,
                          clusteringInference,
                          predictedClusterId,
                          rawClusteringAccuracy,
                          rollingClusteringAccuracy,
                          clusterHomogeneity,
                          clusteringConfidence):
  trace['clusteringInference'].append(clusteringInference)
  trace['predictedClusterId'].append(predictedClusterId)
  trace['rawClusteringAccuracy'].append(rawClusteringAccuracy)
  trace['rollingClusteringAccuracy'].append(rollingClusteringAccuracy)
  trace['clusterHomogeneity'].append(clusterHomogeneity)
  trace['clusteringConfidence'].append(clusteringConfidence)
  return trace



def updateTrace(trace,
                recordNumber,
                sensorValue,
                actualCategory,
                tmActiveCells,
                tmPredictedActiveCells,
                rawAnomalyScore,
                rollingAnomalyScore,
                tpActiveCells,
                classificationInference,
                rawClassificationAccuracy,
                rollingClassificationAccuracy):
  trace['recordNumber'].append(recordNumber)
  trace['sensorValue'].append(sensorValue)
  trace['actualCategory'].append(actualCategory)
  trace['tmActiveCells'].append(tmActiveCells)
  trace['tmPredictedActiveCells'].append(tmPredictedActiveCells)
  trace['rawAnomalyScore'].append(rawAnomalyScore)
  trace['rollingAnomalyScore'].append(rollingAnomalyScore)
  trace['tpActiveCells'].append(tpActiveCells)
  trace['classificationInference'].append(classificationInference)
  trace['rawClassificationAccuracy'].append(rawClassificationAccuracy)
  trace['rollingClassificationAccuracy'].append(rollingClassificationAccuracy)
  return trace



def outputClusteringInfo(clusteringInference,
                         predictedClusterId,
                         clusteringAccuracy,
                         clusterHomogeneity,
                         clusteringConfidence,
                         numClusters):
  # Clustering
  _LOGGER.debug('-> clusteringInference: %s' % clusteringInference)
  _LOGGER.debug('-> predictedClusterId: %s' % predictedClusterId)
  _LOGGER.debug('-> clusteringAccuracy: %s / 1' % clusteringAccuracy)
  _LOGGER.debug('-> clusterHomogeneity: %s / 100' % clusterHomogeneity)
  _LOGGER.debug('-> clusteringConfidence: %s' % clusteringConfidence)
  _LOGGER.debug('-> numClusters: %s' % numClusters)



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



def outputInterClusterDist(clustering, numCells):
  if _LOGGER.getEffectiveLevel() == logging.DEBUG:
    interClusterDist = interClusterDistances(clustering.getClusters(),
                                             clustering.getNewCluster(), 
                                             numCells)
    _LOGGER.debug('-> inter-cluster distances: %s' % interClusterDist)



def outputClustersStructure(clustering):
  if _LOGGER.getEffectiveLevel() == logging.DEBUG:
    labelClusters(clustering)

    # sort cluster-category frequencies by label and cumulative number of 
    # points
    sortedFreqDicts = sorted(
      clustering.clusterActualCategoriesFrequencies(),
      key=lambda x: (clustering.getClusterById(x['clusterId']).getLabel(),
                     sum([freq['numberOfPoints']
                          for freq in x['actualCategoryFrequencies']])))

    for frequencyDict in sortedFreqDicts:
      clusterId = frequencyDict['clusterId']
      actualCategoryFrequencies = frequencyDict['actualCategoryFrequencies']
      cluster = clustering.getClusterById(clusterId)
      _LOGGER.debug('-> frequencies of actual categories in cluster %s.'
                    % clusterId)
      _LOGGER.debug('-> cluster info: %s' % cluster)
      for freq in actualCategoryFrequencies:
        _LOGGER.debug('* actualCategory: %s' % freq['actualCategory'])
        _LOGGER.debug('* numberPoints: %s' % freq['numberOfPoints'])
        _LOGGER.debug('')



def getNetworkSetup(networkConfig):
  networkSetup = {}

  spEnabled = networkConfig['spRegionConfig'].get(
    'regionEnabled')
  tmEnabled = networkConfig['tmRegionConfig'].get(
    'regionEnabled')
  tpEnabled = networkConfig['tpRegionConfig'].get(
    'regionEnabled')
  classifierType = networkConfig['classifierRegionConfig'].get(
    'regionType')

  cells = networkConfig['tmRegionConfig']['regionParams']['cellsPerColumn']
  columns = networkConfig['tmRegionConfig']['regionParams']['columnCount']

  networkSetup['spEnabled'] = spEnabled
  networkSetup['tmEnabled'] = tmEnabled
  networkSetup['tpEnabled'] = tpEnabled
  networkSetup['classifierType'] = classifierType
  networkSetup['numTmCells'] = cells * columns
  return networkSetup



def generateExpId(filePath, networkSetup):
  baseName = filePath.split('/')[-1].split('.csv')[0]
  return '%s_sp=%s_tm=%s_tp=%s_%s' % (baseName,
                                      networkSetup['spEnabled'],
                                      networkSetup['tmEnabled'],
                                      networkSetup['tpEnabled'],
                                      networkSetup['classifierType'][3:-6])



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



def runNetwork(networkConfig, filePath, runClustering):
  dataSource = FileRecordStream(streamID=filePath)
  network = configureNetwork(dataSource, networkConfig)

  (sensorRegion,
   spRegion,
   tmRegion,
   tpRegion,
   classifierRegion) = enableRegionLearning(network, networkConfig)

  trace = initTrace(runClustering)
  numCells = networkConfig['tmRegionConfig']['regionParams']['inputWidth'] * \
             networkConfig['tmRegionConfig']['regionParams']['cellsPerColumn']
  if runClustering:
    clustering = Clustering(numCells,
                            MERGE_THRESHOLD,
                            ANOMALOUS_THRESHOLD,
                            STABLE_THRESHOLD,
                            MIN_CLUSTER_SIZE,
                            SIMILARITY_THRESHOLD)

  recordNumber = 0
  while 1:
    try:
      network.run(1)

      (sensorValue,
       actualCategory,
       tmActiveCells,
       tmPredictedActiveCells,
       rawAnomalyScore,
       rollingAnomalyScore,
       tpActiveCells,
       classificationInference,
       rawClassificationAccuracy,
       rollingClassificationAccuracy) = computeNetworkStats(trace,
                                                            ROLLING_ACCURACY_WINDOW,
                                                            sensorRegion,
                                                            tmRegion,
                                                            tpRegion,
                                                            classifierRegion)

      trace = updateTrace(trace,
                          recordNumber,
                          sensorValue,
                          actualCategory,
                          tmActiveCells,
                          tmPredictedActiveCells,
                          rawAnomalyScore,
                          rollingAnomalyScore,
                          tpActiveCells,
                          classificationInference,
                          rawClassificationAccuracy,
                          rollingClassificationAccuracy)

      if recordNumber % 500 == 0:
        outputClassificationInfo(recordNumber,
                                 sensorValue,
                                 actualCategory,
                                 rollingAnomalyScore,
                                 classificationInference,
                                 rollingClassificationAccuracy)

      if runClustering:
        if CELLS_TO_CLUSTER == 'tmActiveCells':
          tmCells = tmActiveCells
        elif CELLS_TO_CLUSTER == 'tmPredictedActiveCells':
          tmCells = tmPredictedActiveCells
        else:
          raise ValueError(
            'CELLS_TO_CLUSTER value can only be "tmActiveCells" '
            'or "tmPredictedActiveCells" but is %s'
            % CELLS_TO_CLUSTER)

        if ANOMALY_SCORE == 'rollingAnomalyScore':
          anomalyScore = rollingAnomalyScore
        elif ANOMALY_SCORE == 'rawAnomalyScore':
          anomalyScore = rawAnomalyScore
        else:
          raise ValueError('ANOMALY_SCORE value can only be "rawAnomalyScore" '
                           'or "rollingAnomalyScore" but is %s'
                           % ANOMALY_SCORE)
        (predictedCluster,
         clusteringConfidence) = clustering.cluster(recordNumber,
                                                    tmCells,
                                                    anomalyScore,
                                                    actualCategory)
        (clusteringInference,
         predictedClusterId,
         rawClusteringAccuracy,
         rollingClusteringAccuracy,
         clusterHomogeneity) = computeClusteringStats(trace,
                                                      predictedCluster,
                                                      clustering,
                                                      actualCategory)
        trace = updateClusteringTrace(trace,
                                      clusteringInference,
                                      predictedClusterId,
                                      rawClusteringAccuracy,
                                      rollingClusteringAccuracy,
                                      clusterHomogeneity,
                                      clusteringConfidence)
        if recordNumber % 100 == 0:
          numClusters = len(clustering.getClusters())
          outputClusteringInfo(clusteringInference,
                               predictedClusterId,
                               rollingClusteringAccuracy,
                               clusterHomogeneity,
                               clusteringConfidence,
                               numClusters)

      recordNumber += 1
    except StopIteration:
      print "Data streaming completed!"
      break

  if runClustering:
    outputClustersStructure(clustering)
    outputInterClusterDist(clustering, numCells)
  return trace, recordNumber



def computeClusteringStats(trace,
                           predictedCluster,
                           clustering,
                           actualCategory):
  (clusteringInference,
   predictedClusterId,
   clusterHomogeneity) = getClusteringInference(predictedCluster, clustering)

  # If clustering inference is None (meaning the data is not stable) then 
  # rawClusteringAccuracy will be None as well
  rawClusteringAccuracy = computeAccuracy(clusteringInference, actualCategory)

  rollingClusteringAccuracy = movingAverage(trace,
                                            'rollingClusteringAccuracy',
                                            ROLLING_ACCURACY_WINDOW,
                                            rawClusteringAccuracy,
                                            actualCategory,
                                            IGNORE_NOISE)
  return (clusteringInference,
          predictedClusterId,
          rawClusteringAccuracy,
          rollingClusteringAccuracy,
          clusterHomogeneity)



def computeNetworkStats(trace,
                        rollingWindowSize,
                        sensorRegion,
                        tmRegion,
                        tpRegion,
                        classifierRegion):
  """ 
  Compute HTM network statistics 
  """
  sensorValue = sensorRegion.getOutputData('sourceOut')[0]
  actualCategory = sensorRegion.getOutputData('categoryOut')[0]

  if tmRegion:
    tmPredictedActiveCells = tmRegion.getOutputData(
      'predictedActiveCells').astype(int)
    tmActiveCells = tmRegion.getOutputData('activeCells').astype(int)
    rawAnomalyScore = tmRegion.getOutputData('anomalyScore')[0]
    rollingAnomalyScore = movingAverage(trace,
                                        'rollingAnomalyScore',
                                        rollingWindowSize,
                                        rawAnomalyScore,
                                        actualCategory,
                                        IGNORE_NOISE)
  else:
    tmActiveCells = None
    tmPredictedActiveCells = None
    rawAnomalyScore = None
    rollingAnomalyScore = None

  if tpRegion:
    tpActiveCells = tpRegion.getOutputData('mostActiveCells')
    tpActiveCells = tpActiveCells.nonzero()[0]
  else:
    tpActiveCells = None

  classificationInference = getClassifierInference(classifierRegion)
  rawClassificationAccuracy = computeAccuracy(classificationInference,
                                              actualCategory)

  rollingClassificationAccuracy = movingAverage(trace,
                                                'rollingClassificationAccuracy',
                                                rollingWindowSize,
                                                rawClassificationAccuracy,
                                                actualCategory,
                                                IGNORE_NOISE)

  return (sensorValue,
          actualCategory,
          tmActiveCells,
          tmPredictedActiveCells,
          rawAnomalyScore,
          rollingAnomalyScore,
          tpActiveCells,
          classificationInference,
          rawClassificationAccuracy,
          rollingClassificationAccuracy)



def getClusteringInference(predictedCluster, clustering):
  if clustering is None:
    return None, None, None

  if predictedCluster:
    predictedClusterId = predictedCluster.getId()
    labelClusters(clustering)
    clusteringInference = predictedCluster.getLabel()
  else:
    clusteringInference = None
    predictedClusterId = None

  clusterHomogeneity = computeClusterHomogeneity(clustering)

  return clusteringInference, predictedClusterId, clusterHomogeneity



def computeClusterHomogeneity(clustering):
  numCorrect = 0
  numPoints = 0
  for cluster in clustering.getClusters():
    for point in cluster.getPoints():
      if point.getLabel() == cluster.getLabel():
        numCorrect += 1
      numPoints += 1
  if numPoints > 0:
    return 100.0 * numCorrect / numPoints
  else:
    return 0.0



def labelClusters(clustering):
  for frequencyDict in clustering.clusterActualCategoriesFrequencies():
    actualCategoryFrequencies = frequencyDict['actualCategoryFrequencies']
    clusterId = frequencyDict['clusterId']
    cluster = clustering.getClusterById(clusterId)
    highToLowFreqs = sorted(actualCategoryFrequencies,
                            key=lambda x: -x['numberOfPoints'])
    bestCategory = highToLowFreqs[0]['actualCategory']
    cluster.setLabel(bestCategory)



def runExperiment(networkConfig, inputFilePath, runClustering):
  networkSetup = getNetworkSetup(networkConfig)
  networkTrace, numPoints = runNetwork(networkConfig,
                                       inputFilePath,
                                       runClustering)
  expId = generateExpId(inputFilePath, networkSetup)
  expResult = {
    'expId': expId,
    'expTrace': networkTrace,
    'networkSetup': networkSetup,
    'inputFilePath': inputFilePath,
    'numPoints': numPoints
  }

  return expResult



def saveResults(outFile, expResults, runClustering):
  """
  Save final clustering and classification accuracies to CSV
  :param outFile: (str) path to CSV file where to save data
  :param expResults: (list of dict) experiment results
  """
  headers = ['fileName', 'finalClassificationAccuracy']
  if runClustering:
    headers.append('finalClusteringAccuracy')

  with open(outFile, 'wb') as fw:
    writer = csv.writer(fw)
    writer.writerow(headers)
    for i in range(len(expResults)):
      inputFile = expResults[i]['inputFilePath']
      expTrace = expResults[i]['expTrace']
      classifAccuracy = expTrace['rollingClassificationAccuracy'][-1]
      if runClustering:
        clustAccuracy = expTrace['rollingClusteringAccuracy'][-1]
        writer.writerow([inputFile, classifAccuracy, clustAccuracy])

    _LOGGER.info('Results saved to %s\n' % outFile)



def saveTraces(baseOutFile, expResults):
  """
  Save experiments network traces to CSV
  :param baseOutFile: (str) base name of the output file.
  :param expResults: (list of dict) experiment results
  """

  for expResult in expResults:
    expTrace = expResult['expTrace']
    numPoints = len(expTrace['recordNumber'])
    outFile = baseOutFile % expResult['expId']
    with open(outFile, 'wb') as f:
      writer = csv.writer(f)
      headers = expTrace.keys()
      writer.writerow(headers)
      for i in range(numPoints):
        row = []
        for t in expTrace.keys():
          if len(expTrace[t]) > i:
            if t in ['tmPredictedActiveCells', 'tmActiveCells']:
              row.append(json.dumps(list(expTrace[t][i].nonzero()[0])))
            elif type(expTrace[t][i]) == list:
              row.append(json.dumps(expTrace[t][i]))
            else:
              row.append(expTrace[t][i])
          else:
            row.append(None)
        writer.writerow(row)

    _LOGGER.info('traces saved to: %s' % outFile)



def run(resultsOutputFile,
        tracesOutputFile,
        inputFiles,
        networkConfigsFile,
        plotResults,
        runClustering):
  with open(networkConfigsFile, 'rb') as jsonFile:
    networkConfigurations = simplejson.load(jsonFile)

  expResults = []
  for networkConfig in networkConfigurations:
    for inputFile in inputFiles:
      expResult = runExperiment(networkConfig, inputFile, runClustering)
      expResults.append(expResult)
      if plotResults:
        # traces = loadTraces(fileName)
        traces = expResult['expTrace']
        tmParams = networkConfig['tmRegionConfig']['regionParams']
        numCells = tmParams['cellsPerColumn'] * tmParams['inputWidth']
        numClusters = len(set(traces['actualCategory']))
        outputDir = TRACES_OUTPUT_FILE[:-4] % inputFile.split('/')[-1][:-4]
        if not os.path.exists(outputDir):
          os.makedirs(outputDir)
        cellsType = CELLS_TO_CLUSTER
        numSteps = len(traces['recordNumber'])
        pointsToPlot = numSteps / 10

        vizInterCategoryClusters(traces,
                                 outputDir,
                                 cellsType,
                                 numCells,
                                 pointsToPlot)

        vizInterSequenceClusters(traces, outputDir, cellsType, numCells,
                                 numClusters)

        xl = None
        plotTemporalMemoryStates = False
        title = inputFile.split('/')[-1]
        outputFile = '%s.png' % inputFile[:-4]
        plotTraces(xl, traces, title, outputFile, plotTemporalMemoryStates)

  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  saveTraces(tracesOutputFile, expResults)
  saveResults(resultsOutputFile, expResults, runClustering)



def main():
  dominoStats = {
    "FILE_NAMES": FILE_NAMES,
    "HTM_NETWORK_CONFIGS": HTM_NETWORK_CONFIGS.split('/')[-1],
    "CLUSTERING": CLUSTERING,
    "MERGE_THRESHOLD": MERGE_THRESHOLD,
    "ANOMALOUS_THRESHOLD": ANOMALOUS_THRESHOLD,
    "STABLE_THRESHOLD": STABLE_THRESHOLD,
    "MIN_CLUSTER_SIZE": MIN_CLUSTER_SIZE,
    "SIMILARITY_THRESHOLD": SIMILARITY_THRESHOLD,
    "ROLLING_ACCURACY_WINDOW": ROLLING_ACCURACY_WINDOW,
    "CELLS_TO_CLUSTER": CELLS_TO_CLUSTER,
    "IGNORE_NOISE": IGNORE_NOISE,
    "ANOMALY_SCORE": ANOMALY_SCORE
  }

  with open('dominostats.json', 'wb') as f:
    f.write(json.dumps(dominoStats))

  run(RESULTS_OUTPUT_FILE,
      TRACES_OUTPUT_FILE,
      INPUT_FILES,
      HTM_NETWORK_CONFIGS,
      PLOT_RESULTS,
      CLUSTERING)



if __name__ == '__main__':
  main()
