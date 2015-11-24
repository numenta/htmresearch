import time
from htmresearch.frameworks.nlp.runner import Runner
from htmresearch.frameworks.nlp.htm_runner import HTMRunner
from htmresearch.support.network_text_data_generator import NetworkDataGenerator
from matplotlib import pyplot as plt
plt.ion()

import htmresearch
import nupic
import csv
import numpy as np

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

def calculate_classificastion_accuracy(categoryLabel, tmCellUnion, knnInLastNSequences=20, knn_number=3):
  predicted_label = []
  actual_label = []
  for seqID in xrange(knn_number+1, len(categoryLabel)):
    tmCellsdr = tmCellUnion[seqID]
    knn_prediction = knn_classifier(tmCellsdr, tmCellUnion[:seqID-1], categoryLabel, knnInLastNSequences, knn_number)
    predicted_label.append(knn_prediction)
    actual_label.append(categoryLabel[seqID])

  accuracy = (np.array(predicted_label) == np.array(actual_label)).astype('float32')
  return accuracy


def knn_classifier(tmCellsdr, tmCellUnion, categoryLabel, knnInLastNSequences=20, knn_number=3):
  """
  Run knn classifier on the last observed [knnInLastNSequences] elements, with n=knn_number
  :param tmCellsdr: SDR to be classified
  :param tmCellUnion: List of SDRs
  :param categoryLabel: List of category labels
  :param knnInLastNSequences:
  :param knn_number:
  :return:
  """
  numSample = len(tmCellUnion)
  if numSample > knn_number+1:
    overlap_score = []
    for seqID2 in xrange(max(numSample-knnInLastNSequences, 0), numSample-1):
      overlap_score.append(np.sum(np.logical_and(tmCellUnion[seqID2], tmCellsdr)))

    sortidx = np.argsort(overlap_score)
    best_match = np.array(range(max(numSample - knnInLastNSequences, 0), numSample-1))[sortidx[-(knn_number + 1):-1]]
    categorylabel_values = np.array(categoryLabel)
    predicted_label = np.argmax(np.bincount(categorylabel_values[best_match]))
  else:
    # make a random guess if the numSample < knn_number
    predicted_label = np.random.randint(2)
  return predicted_label

def get_animal_vegetable_list():
  animal_reader = csv.reader(open('data/animal_eat_vegetables/animals.txt', 'r'))
  animals = []
  for row in animal_reader:
    animals.append(row[0])

  vegetable_reader = csv.reader(open('data/animal_eat_vegetables/vegetables.txt', 'r'))
  vegetables = []
  for row in vegetable_reader:
    vegetables.append(row[0])
  return animals, vegetables

def get_nupic_regions(network):
  sensorRegion = None
  spRegion = None
  tmRegion = None
  tpRegion = None
  knnRegion = None
  for region in network.regions.values():
    regionInstance = region
    if type(regionInstance.getSelf()) is htmresearch.regions.LanguageSensor.LanguageSensor:
      sensorRegion = regionInstance.getSelf()
    elif type(regionInstance.getSelf()) is nupic.regions.TPRegion.TPRegion:
      tmRegion = regionInstance.getSelf()
    elif type(regionInstance.getSelf()) is htmresearch.regions.TemporalPoolerRegion.TemporalPoolerRegion:
      tpRegion = regionInstance.getSelf()
    elif type(regionInstance.getSelf()) is nupic.regions.KNNClassifierRegion.KNNClassifierRegion:
      knnRegion = regionInstance.getSelf()
    elif type(regionInstance.getSelf()) is nupic.regions.SPRegion.SPRegion:
      spRegion = regionInstance.getSelf()

  return sensorRegion, spRegion, tmRegion, tpRegion, knnRegion


def movingAverage(data, window):
  movingAverage = []

  for i in xrange(len(data)):
    start = max(0, i - window)
    end = i+1
    values = data[start:end]
    movingAverage.append(sum(values) / float(len(values)))

  return movingAverage


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

  # args = inputParameters(retina="en_associative_64_univ",
  #                        apiKey="7c164cd0-fca0-11e3-80ac-f7122a45615d",
  #                        networkConfigPath='data/network_configs/tm_knn_4k_retina.json')

  args = inputParameters(retina="en_associative_64_univ",
                         apiKey="7c164cd0-fca0-11e3-80ac-f7122a45615d",
                         networkConfigPath='data/network_configs/tm_tp_knn_4k_retina.json')

  # setup HTM Runner
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

  sensorRegion, spRegion, tmRegion, tpRegion, knnRegion = get_nupic_regions(runner.model.network)
  tmRegion.learningMode = True
  tmRegion.computePredictedActiveCellIndices = True

  tpRegion._pooler._globalInhibition = True
  # i = 0
  # runner.resetModel(i)
  # if runner.verbosity > 0:
  #   print "\tTraining and testing for run {}.".format(i)


  animals, vegetables = get_animal_vegetable_list()
  vegetable = {}
  animal = {}
  tmCellUnion = []
  tmInputUnion = []
  tpOutput = []
  categoryLabel = []

  categoryList = ['animal-eats-vegetable', 'vegetable-eats-animal']

  knnInLastNSequences = 20
  knn_number = 1

  plt.close('all')
  plt.figure(1)
  plt.show()

  accuracy = []
  accuracy_tp = []
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
        tpRegionInput = {"activeCells": tmRegionOutput["bottomUpOut"],
                         "predictedActiveCells": tmRegionOutput["predictedActiveCells"],
                         "sequenceIdIn": sensorOutput["sequenceIdOut"],
                         "resetIn": sensorOutput['resetOut']}
        tpRegionOutputs = {"mostActiveCells": np.zeros((tpRegion._columnCount,))}
        tpRegion.compute(tpRegionInput, tpRegionOutputs)

      if word > 0:
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
    predicted_label = knn_classifier(tmCellActivation,
                                         tmCellUnion,
                                         categoryLabel,
                                         knnInLastNSequences=knnInLastNSequences,
                                         knn_number=knn_number)
    accuracy.append(sensorOutput['categoryOut'][0]==predicted_label)
    if tpRegion is not None:
      predicted_label_tp = knn_classifier(tpRegionOutputs["mostActiveCells"],
                                           tpOutput,
                                           categoryLabel,
                                           knnInLastNSequences=knnInLastNSequences,
                                           knn_number=knn_number)
      accuracy_tp.append(sensorOutput['categoryOut'][0]==predicted_label_tp)

    print " sequence: ", sensorOutput['sequenceIdOut'][0], \
          " category: ", categoryList[sensorOutput['categoryOut'][0]], \
          " predicted category: ", categoryList[predicted_label]

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
      plt.plot(movingAverage(accuracy_tp, winLen), color='red')
      plt.ylabel(' Classification Accuracy ')
      plt.xlabel(' Training Samples #')
      plt.ylim([.5, 1.0])
      plt.draw()
      time.sleep(0.1)


  categoryLabel = np.array(categoryLabel)

  # evaluate cortical.io sdr overlaps
  vegetable_sdrs = vegetable.values()
  animal_sdrs = animal.values()
  evaluateSDROverlap(vegetable_sdrs, animal_sdrs)


  # plot accuracy over time
  accuracy_control = calculate_classificastion_accuracy(categoryLabel, tmInputUnion)
  accuracy = calculate_classificastion_accuracy(categoryLabel, tmCellUnion, knnInLastNSequences=30, knn_number=3)
  print " mean accuracy: ", np.mean(accuracy[-100:])
  winLen = 50
  accuracy_average = movingAverage(accuracy, winLen)
  accuracy_control_average = movingAverage(accuracy_control, winLen)

  plt.figure()
  plt.plot(accuracy_average)
  plt.plot(accuracy_control_average)
  plt.legend(['Classification with predicted TM cells', 'with bag of SDRs'], loc=4)
  plt.ylabel(' Classification Accuracy ')
  plt.xlabel(' Training Samples #')
  plt.ylim([.0, 1.0])


  # evaluate overlap on TP representations
  group1 = np.where(categoryLabel==0)[0]
  group2 = np.where(categoryLabel==1)[0]

  group1 = group1[100:]
  group2 = group2[100:]

  dist_within_category = []
  dist_across_category = []
  maximum_dist_across_category = []
  maximum_dist_within_category = []
  for i in group1:
    dist = []
    for j in group2:
      if i-10 < j < i:
        dist.append(np.sum(np.logical_and(tmCellUnion[i], tmCellUnion[j])))
    if len(dist) > 0:
      maximum_dist_across_category.append(max(dist))
      dist_across_category += dist

    dist = []
    for j in group1:
      if i-10 < j < i:
        dist.append(np.sum(np.logical_and(tmCellUnion[i], tmCellUnion[j])))
    if len(dist) > 0:
      maximum_dist_within_category.append(max(dist))
      dist_within_category += dist

  maximum_dist_across_category = np.array(maximum_dist_across_category)
  maximum_dist_within_category = np.array(maximum_dist_within_category)
  print "Overlap of TP representation within category", np.mean(dist_within_category)
  print "Overlap of TP representation across category", np.mean(dist_across_category)


  hist_bins = np.linspace(0, np.max(dist_within_category+dist_across_category), 20)
  hist_bins_center = (hist_bins[1:] + hist_bins[:-1])/2

  overlap_within_category_dist = np.histogram(dist_within_category, bins=hist_bins)
  overlap_within_category_dist = overlap_within_category_dist[0].astype('float32')/sum(overlap_within_category_dist[0])

  overlap_across_category_dist = np.histogram(dist_across_category, bins=hist_bins)
  overlap_across_category_dist = overlap_across_category_dist[0].astype('float32')/sum(overlap_across_category_dist[0])

  plt.figure()
  plt.subplot(3,1,1)
  plt.plot(hist_bins_center, overlap_across_category_dist)
  plt.plot(hist_bins_center, overlap_within_category_dist)
  plt.xlabel(' Overlap of TP representations')
  plt.legend(['across category', 'within category'])

  overlap_within_category_dist = np.histogram(maximum_dist_within_category, bins=hist_bins)
  overlap_within_category_dist = overlap_within_category_dist[0].astype('float32')/sum(overlap_within_category_dist[0])

  overlap_across_category_dist = np.histogram(maximum_dist_across_category, bins=hist_bins)
  overlap_across_category_dist = overlap_across_category_dist[0].astype('float32')/sum(overlap_across_category_dist[0])

  plt.subplot(3,1,2)
  plt.plot(hist_bins_center, overlap_across_category_dist)
  plt.plot(hist_bins_center, overlap_within_category_dist)
  plt.xlabel(' Overlap with nearest neighbor')
  plt.legend(['across category', 'within category'])

  plt.subplot(3,1,3)
  plt.hist(maximum_dist_within_category - maximum_dist_across_category )
  plt.xlabel(' Overlap with nearest neighbor (within - across)')


  # calculate number of predicted cells over time
  nOnBits = []
  for i in xrange(len(tmCellUnion)):
    nOnBits.append(np.sum(tmCellUnion[i]))

  plt.figure()
  plt.plot(nOnBits)
  plt.ylabel(' Predicted Cells #')
  plt.xlabel(' Training Samples #')