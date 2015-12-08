#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

"""
Script to run a simple sentence classification experiment with HTM temporal memory
The goal is to classify two class of sentences:
"The animal eats vegetable" (e.g. The rabbit eats carrot)
and
"The vegetable eats animal" (e.g. The carrot eats rabbit)
"""

import time
import csv
import argparse

from htmresearch.frameworks.nlp.htm_runner import HTMRunner
from htmresearch.support.network_text_data_generator import NetworkDataGenerator
from htmresearch.regions.TemporalPoolerRegion import TemporalPoolerRegion
from htmresearch.regions.LanguageSensor import  LanguageSensor
import nupic
import numpy as np

from matplotlib import pyplot as plt
plt.ion()

def getNupicRegions(network):
  sensorRegion = None
  spRegion = None
  tmRegion = None
  tpRegion = None
  knnRegion = None
  for region in network.regions.values():
    regionInstance = region
    if type(regionInstance.getSelf()) is LanguageSensor:
      sensorRegion = regionInstance.getSelf()
    elif type(regionInstance.getSelf()) is nupic.regions.TPRegion.TPRegion:
      tmRegion = regionInstance.getSelf()
    elif type(regionInstance.getSelf()) is TemporalPoolerRegion:
      tpRegion = regionInstance.getSelf()
    elif type(regionInstance.getSelf()) is nupic.regions.KNNClassifierRegion.KNNClassifierRegion:
      knnRegion = regionInstance.getSelf()
    elif type(regionInstance.getSelf()) is nupic.regions.SPRegion.SPRegion:
      spRegion = regionInstance.getSelf()

  return sensorRegion, spRegion, tmRegion, tpRegion, knnRegion


def getAnimalVegetableList():
  animal_reader = csv.reader(open('data/animal_eat_vegetables/animals.txt', 'r'))
  animals = []
  for row in animal_reader:
    animals.append(row[0])

  vegetable_reader = csv.reader(open('data/animal_eat_vegetables/vegetables.txt', 'r'))
  vegetables = []
  for row in vegetable_reader:
    vegetables.append(row[0])
  return animals, vegetables


def movingAverage(data, window):
  movingAverage = []

  for i in xrange(len(data)):
    start = max(0, i - window)
    end = i+1
    values = data[start:end]
    movingAverage.append(sum(values) / float(len(values)))

  return movingAverage


def evaluateSDROverlap(vegetable_sdrs, animal_sdrs):
  """
  Evaluate overlap between SDRs for the vegetable category and animal category
  :param vegetable_sdrs: List of numpy arrays
  :param animal_sdrs: List of numpy arrays
  :return:
  """
  overlap_vegetable_vegetable = []
  for i in xrange(len(vegetable_sdrs)):
    for j in xrange(i+1, len(vegetable_sdrs)):
      overlap_vegetable_vegetable.append(np.sum(np.logical_and(vegetable_sdrs[i], vegetable_sdrs[j])))

  overlap_animal_animal = []
  for i in xrange(len(animal_sdrs)):
    for j in xrange(i+1, len(animal_sdrs)):
      overlap_animal_animal.append(np.sum(np.logical_and(animal_sdrs[i], animal_sdrs[j])))

  overlap_vegetable_animal = []
  for i in xrange(len(vegetable_sdrs)):
    for j in xrange(len(animal_sdrs)):
      overlap_vegetable_animal.append(np.sum(np.logical_and(vegetable_sdrs[i], animal_sdrs[j])))

  hist_bins = np.linspace(0, np.max(overlap_animal_animal+overlap_vegetable_vegetable), 15)
  hist_bins_center = (hist_bins[1:] + hist_bins[:-1])/2
  overlap_vegetable_animal_dist = np.histogram(overlap_vegetable_animal, bins=hist_bins)
  overlap_vegetable_animal_dist = overlap_vegetable_animal_dist[0].astype('float32')/sum(overlap_vegetable_animal_dist[0])

  overlap_animal_animal_dist = np.histogram(overlap_animal_animal, bins=hist_bins)
  overlap_animal_animal_dist = overlap_animal_animal_dist[0].astype('float32')/sum(overlap_animal_animal_dist[0])

  overlap_vegetable_vegetable_dist = np.histogram(overlap_vegetable_vegetable, bins=hist_bins)
  overlap_vegetable_vegetable_dist = overlap_vegetable_vegetable_dist[0].astype('float32')/sum(overlap_vegetable_vegetable_dist[0])

  plt.figure()
  plt.plot(hist_bins_center, overlap_animal_animal_dist)
  plt.plot(hist_bins_center, overlap_vegetable_vegetable_dist)
  plt.plot(hist_bins_center, overlap_vegetable_animal_dist)
  plt.legend(['animal-animal', 'vegetable-vegetable', 'animal-vegetable'])
  plt.axvline(x=np.mean(overlap_animal_animal), color='blue', linestyle='dashed')
  plt.axvline(x=np.mean(overlap_vegetable_vegetable), color='green', linestyle='dashed')
  plt.axvline(x=np.mean(overlap_vegetable_animal), color='red', linestyle='dashed')
  plt.xlabel(' Overlap ')

  print " overlap animal-animal mean: ", np.mean(overlap_animal_animal), " max: ", np.max(overlap_animal_animal)
  print " overlap vegetable-vegetable", np.mean(overlap_vegetable_vegetable), " max: ", np.max(overlap_vegetable_vegetable)
  print " overlap animal-vegetable", np.mean(overlap_vegetable_animal), " max: ", np.max(overlap_vegetable_animal)


def calculateClassificastionAccuracy(categoryLabel, tmCellUnion, knnInLastNSequences=20, knnNumber=3):
  predicted_label = []
  actual_label = []
  for seqID in xrange(knnNumber+1, len(categoryLabel)):
    tmCellsdr = tmCellUnion[seqID]
    knn_prediction = knnClassifier(tmCellsdr, tmCellUnion[:seqID - 1], categoryLabel, knnInLastNSequences, knnNumber)
    predicted_label.append(knn_prediction)
    actual_label.append(categoryLabel[seqID])

  accuracy = (np.array(predicted_label) == np.array(actual_label)).astype('float32')
  return accuracy


def knnClassifier(tmCellsdr, tmCellUnion, categoryLabel, knnInLastNSequences=20, knnNumber=3):
  """
  Run knn classifier on the last observed [knnInLastNSequences] elements, with n=knnNumber
  :param tmCellsdr: SDR to be classified
  :param tmCellUnion: List of SDRs
  :param categoryLabel: List of category labels
  :param knnInLastNSequences:
  :param knnNumber:
  :return:
  """
  numSample = len(tmCellUnion)
  if numSample > knnNumber+1:
    overlap_score = []
    for seqID2 in xrange(max(numSample-knnInLastNSequences, 0), numSample-1):
      overlap_score.append(np.sum(np.logical_and(tmCellUnion[seqID2], tmCellsdr)))

    sortidx = np.argsort(overlap_score)
    best_match = np.array(range(max(numSample - knnInLastNSequences, 0), numSample-1))[sortidx[-(knnNumber + 1):-1]]
    categorylabel_values = np.array(categoryLabel)
    predicted_label = np.argmax(np.bincount(categorylabel_values[best_match]))
  else:
    # make a random guess if the numSample < knnNumber
    predicted_label = np.random.randint(2)
  return predicted_label


def plotAccuracyOverTime(categoryLabel, tmCellUnion, tmInputUnion):
  accuracyControl = calculateClassificastionAccuracy(categoryLabel, tmInputUnion)
  accuracy = calculateClassificastionAccuracy(categoryLabel, tmCellUnion, knnInLastNSequences=30, knnNumber=3)
  print " mean accuracy: ", np.mean(accuracy[-100:])
  winLen = 50
  accuracyAverage = movingAverage(accuracy, winLen)
  accuracyControlAverage = movingAverage(accuracyControl, winLen)

  plt.figure()
  plt.plot(accuracyAverage)
  plt.plot(accuracyControlAverage)
  plt.legend(['Classification with predicted TM cells', 'with bag of SDRs'], loc=4)
  plt.ylabel(' Classification Accuracy ')
  plt.xlabel(' Training Samples #')
  plt.ylim([.0, 1.0])


def evaulateTPrepresentationOverlap(categoryLabel, tmCellUnion, startFrom=50):
  # evaluate overlap on TP representations
  group1 = np.where(categoryLabel==0)[0]
  group2 = np.where(categoryLabel==1)[0]

  group1 = group1[startFrom:]
  group2 = group2[startFrom:]

  overlapWithinCategory = []
  overlapAcrossCategory = []
  maximumOverlapAcrossCategory = []
  maximumOverlapWithinCategory = []
  for i in group1:
    dist = []
    for j in group2:
      if i-10 < j < i:
        dist.append(np.sum(np.logical_and(tmCellUnion[i], tmCellUnion[j])))
    if len(dist) > 0:
      maximumOverlapAcrossCategory.append(max(dist))
      overlapAcrossCategory += dist

    dist = []
    for j in group1:
      if i-10 < j < i:
        dist.append(np.sum(np.logical_and(tmCellUnion[i], tmCellUnion[j])))
    if len(dist) > 0:
      maximumOverlapWithinCategory.append(max(dist))
      overlapWithinCategory += dist

  maximumOverlapAcrossCategory = np.array(maximumOverlapAcrossCategory)
  maximumOverlapWithinCategory = np.array(maximumOverlapWithinCategory)
  print "Overlap of TP representation within category", np.mean(overlapWithinCategory)
  print "Overlap of TP representation across category", np.mean(overlapAcrossCategory)


  histBins = np.linspace(0, np.max(overlapWithinCategory + overlapAcrossCategory), 20)
  histBinsCenter = (histBins[1:] + histBins[:-1]) / 2

  overlapWithinCategoryDistribution = np.histogram(overlapWithinCategory, bins=histBins)
  overlapWithinCategoryDistribution = overlapWithinCategoryDistribution[0].astype('float32') / sum(overlapWithinCategoryDistribution[0])

  overlapAcrossCategoryDistribution = np.histogram(overlapAcrossCategory, bins=histBins)
  overlapAcrossCategoryDistribution = overlapAcrossCategoryDistribution[0].astype('float32') / sum(overlapAcrossCategoryDistribution[0])

  plt.figure()
  plt.subplot(3,1,1)
  plt.plot(histBinsCenter, overlapAcrossCategoryDistribution)
  plt.plot(histBinsCenter, overlapWithinCategoryDistribution)
  plt.xlabel(' Overlap of TP representations')
  plt.legend(['across category', 'within category'])

  overlapWithinCategoryDistribution = np.histogram(maximumOverlapWithinCategory, bins=histBins)
  overlapWithinCategoryDistribution = overlapWithinCategoryDistribution[0].astype('float32') / sum(overlapWithinCategoryDistribution[0])

  overlapAcrossCategoryDistribution = np.histogram(maximumOverlapAcrossCategory, bins=histBins)
  overlapAcrossCategoryDistribution = overlapAcrossCategoryDistribution[0].astype('float32') / sum(overlapAcrossCategoryDistribution[0])

  plt.subplot(3,1,2)
  plt.plot(histBinsCenter, overlapAcrossCategoryDistribution)
  plt.plot(histBinsCenter, overlapWithinCategoryDistribution)
  plt.xlabel(' Overlap with nearest neighbor')
  plt.legend(['across category', 'within category'])

  plt.subplot(3,1,3)
  plt.hist(maximumOverlapWithinCategory - maximumOverlapAcrossCategory)
  plt.xlabel(' Overlap with nearest neighbor (within - across)')


class inputParameters(object):
  def __init__(self,
               dataPath='data/animal_eat_vegetables/animal_eat_vegetable_network.csv',
               networkConfigPath='data/network_configs/tm_knn.json',
               resultsDir='results/',
               experimentName='SimpleClassification',
               experimentType='k-folds',
               loadPath=None,
               modelName='HTMNetwork',
               retinaScaling=1.0,
               retina='en_associative',
               apiKey=None,
               numClasses=1,
               plots=0,
               orderedSplit=True,
               folds=5,
               trainSizes=[50],
               verbosity=1,
               generateData=False,
               votingMethod='last',
               classificationFile='data/animal_eat_vegetables/animal_eat_vegetable.json',
               textPreprocess=False,
               seed=42
                 ):

      self.dataPath = dataPath
      self.networkConfigPath = networkConfigPath
      self.resultsDir = resultsDir
      self.experimentName = experimentName
      self.experimentType = experimentType
      self.loadPath = loadPath
      self.modelName = modelName
      self.retinaScaling = retinaScaling
      self.retina = retina
      self.apiKey = apiKey
      self.numClasses = numClasses
      self.plots = plots
      self.orderedSplit = orderedSplit
      self.folds = folds
      self.trainSizes = trainSizes
      self.verbosity = verbosity
      self.generateData = generateData
      self.votingMethod = votingMethod
      self.classificationFile = classificationFile
      self.textPreprocess = textPreprocess
      self.seed = seed


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--useTPregion",
                      default=0,
                      type=int,
                      help="0 for using direct union of TM cells, 1 for using TP region")
  parser.add_argument("--retina",
                      default="en_associative_64_univ",
                      type=str,
                      help="Name of Cortical.io retina.")
  parser.add_argument("--apiKey",
                      default=None,
                      type=str,
                      help="Key for Cortical.io API (use special key for 4K retina).")
  args = parser.parse_args()

  if args.useTPregion:
    args = inputParameters(retina=args.retina,
                           apiKey=args.apiKey,
                           networkConfigPath='data/network_configs/tm_tp_knn_4k_retina.json')
  else:
    args = inputParameters(retina=args.retina,
                           apiKey=args.apiKey,
                           networkConfigPath='data/network_configs/tm_knn_4k_retina.json')

  runner = HTMRunner(dataPath=args.dataPath,
                     networkConfigPath=args.networkConfigPath,
                     resultsDir=args.resultsDir,
                     experimentName=args.experimentName,
                     experimentType=args.experimentType,
                     loadPath=args.loadPath,
                     modelName=args.modelName,
                     retinaScaling=args.retinaScaling,
                     retina=args.retina,
                     apiKey=args.apiKey,
                     numClasses=args.numClasses,
                     plots=args.plots,
                     orderedSplit=args.orderedSplit,
                     folds=args.folds,
                     trainSizes=args.trainSizes,
                     verbosity=args.verbosity,
                     generateData=args.generateData,
                     votingMethod=args.votingMethod,
                     classificationFile=args.classificationFile,
                     seed=args.seed)
  runner.initModel(0)
  runner.setupData(args.textPreprocess)
  runner.encodeSamples()
  runner.partitionIndices(args.seed)

  sensorRegion, spRegion, tmRegion, tpRegion, knnRegion = getNupicRegions(runner.model.network)
  tmRegion.learningMode = True
  tmRegion.computePredictedActiveCellIndices = True

  categoryList = ['animal-eats-vegetable', 'vegetable-eats-animal']
  animals, vegetables = getAnimalVegetableList()
  vegetable = {}
  animal = {}
  tmCellUnion = []
  tmInputUnion = []
  tpOutput = []
  categoryLabel = []
  accuracy = []
  accuracyTp = []
  knnInLastNSequences = 20
  knnNumber = 1

  plt.close('all')
  plt.figure(1)
  plt.show()
  numTokens = NetworkDataGenerator.getNumberOfTokens(args.dataPath)
  for numSample in xrange(len(numTokens)):
    # union SDR for this sequence
    tmCellActivation = np.zeros((tmRegion._tfdr.cellsPerColumn * tmRegion._tfdr.columnDimensions[0],))
    tmInputActivation = np.zeros((tmRegion._tfdr.columnDimensions[0],))
    print
    for word in xrange(numTokens[numSample]):
      sensorInput = None
      sensorOutput = {'categoryOut': np.array([0]),
                      'resetOut': [None],
                      'sourceOut': None,
                      'sequenceIdOut': [None],
                      'encodingOut': None,
                      'dataOut': np.zeros((sensorRegion.encoder.n, ))}
      sensorRegion.compute(sensorInput, sensorOutput)

      if spRegion is not None:
        spRegionInput = {'bottomUpIn': sensorOutput['dataOut'],
                         'resetIn': sensorOutput['resetOut'],
                         'topDownIn': 0}
        spRegionOutput = {'bottomUpOut': np.zeros((spRegion._sfdr.getNumColumns(), )),
                          'anomalyScore': np.zeros(1),
                          'spatialTopDownOut': 0}
        spRegion.compute(spRegionInput, spRegionOutput)

        tmRegionInput = {'bottomUpIn': spRegionOutput['bottomUpOut'],
                         'resetIn': sensorOutput['resetOut']}
      else:
        tmRegionInput = {'bottomUpIn': sensorOutput['dataOut'],
                         'resetIn': sensorOutput['resetOut']}

      numTMcells = tmRegion._tfdr.cellsPerColumn * tmRegion._tfdr.columnDimensions[0]
      tmRegionOutput = {'bottomUpOut': np.zeros((numTMcells,)).astype('float32'),
                        "activeCells": np.zeros((numTMcells,)).astype('float32'),
                        "predictedActiveCells": np.zeros((numTMcells,)).astype('float32')}
      tmRegion.compute(tmRegionInput, tmRegionOutput)

      if tpRegion is not None:
        resetSignal = sensorOutput['resetOut']
        tpRegionInput = {"activeCells": tmRegionOutput["bottomUpOut"],
                         "predictedActiveCells": tmRegionOutput["predictedActiveCells"],
                         "resetIn": resetSignal}
        tpRegionOutputs = {"mostActiveCells": np.zeros((tpRegion._columnCount,))}
        tpRegion.compute(tpRegionInput, tpRegionOutputs)

      # plain union of TM cells
      # tmCellActivation = np.logical_or(tmCellActivation, tmRegionOutput['bottomUpOut'])
      tmCellActivation = np.logical_or(tmCellActivation, tmRegionOutput["predictedActiveCells"])
      tmInputActivation = np.logical_or(tmInputActivation, tmRegionInput["bottomUpIn"])


      currentWord = sensorOutput['sourceOut']
      print " current word: ", currentWord, \
            " \tTM active inputs: ", np.sum(tmRegionInput['bottomUpIn']), \
            " active cells: ", np.sum(tmRegionOutput['bottomUpOut']), \
            " predicted cells: ", np.sum(tmRegion._tfdr.getPredictedState().reshape(-1).astype('float32')), \
            " predicted-active cells: ", np.sum(tmRegionOutput["predictedActiveCells"]),
      if tpRegion is not None:
        print " tp cells: ", np.sum(tpRegionOutputs["mostActiveCells"])
      else:
        print


      if vegetables.count(currentWord) > 0:
        vegetable[currentWord] = tmRegionInput['bottomUpIn']

      if animals.count(currentWord) > 0:
        animal[currentWord] = tmRegionInput['bottomUpIn']

    # classify sentence
    predictedLabel = knnClassifier(tmCellActivation,
                                   tmCellUnion,
                                   categoryLabel,
                                   knnInLastNSequences=knnInLastNSequences,
                                   knnNumber=knnNumber)
    accuracy.append(sensorOutput['categoryOut'][0] == predictedLabel)
    if tpRegion is not None:
      predicted_label_tp = knnClassifier(tpRegionOutputs["mostActiveCells"],
                                         tpOutput,
                                         categoryLabel,
                                         knnInLastNSequences=knnInLastNSequences,
                                         knnNumber=knnNumber)
      accuracyTp.append(sensorOutput['categoryOut'][0] == predicted_label_tp)

    print " sequence: ", sensorOutput['sequenceIdOut'][0], \
          " category: ", categoryList[sensorOutput['categoryOut'][0]], \
          " predicted category: ", categoryList[predictedLabel]

    sequenceID = sensorOutput['sequenceIdOut'][0]
    tmInputUnion.append(tmInputActivation)
    tmCellUnion.append(tmCellActivation)
    if tpRegion is not None:
      tpOutput.append(tpRegionOutputs["mostActiveCells"])

    categoryLabel.append(sensorOutput['categoryOut'][0])

    if np.mod(numSample, 50) == 0 and numSample > 50:
      winLen = 30
      plt.figure(1)
      plt.plot(movingAverage(accuracy, winLen), color='blue')
      plt.plot(movingAverage(accuracyTp, winLen), color='red')
      plt.ylabel(' Classification Accuracy ')
      plt.xlabel(' Training Samples #')
      plt.ylim([0.0, 1.0])
      plt.draw()
      time.sleep(0.1)


  categoryLabel = np.array(categoryLabel)

  # evaluate cortical.io sdr overlaps
  vegetableSdrs = vegetable.values()
  animalSdrs = animal.values()
  evaluateSDROverlap(vegetableSdrs, animalSdrs)

  plotAccuracyOverTime(categoryLabel, tmCellUnion, tmInputUnion)

  evaulateTPrepresentationOverlap(categoryLabel, tmCellUnion)

  # calculate number of predicted cells over time
  nOnBits = []
  for i in xrange(len(tmCellUnion)):
    nOnBits.append(np.sum(tmCellUnion[i]))
  plt.figure()
  plt.plot(nOnBits)
  plt.ylabel(' Predicted Cells #')
  plt.xlabel(' Training Samples #')