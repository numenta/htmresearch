import numpy as np

from htmresearch.frameworks.clustering.dim_reduction import (project2D,
                                                             projectClusters2D,
                                                             viz2DProjection,
                                                             plotDistanceMat)

from htmresearch.frameworks.clustering.distances import (percentOverlap,
                                                         clusterDist)



def convertNonZeroToSDR(patternNZs, numCells):
  sdrs = []
  for patternNZ in patternNZs:
    sdr = np.zeros(numCells)
    sdr[patternNZ] = 1
    sdrs.append(sdr)

  return sdrs



def assignClusters(traces):
  categories = np.unique(traces['actualCategory'])
  numCategories = len(categories)
  # The noise is labelled as 0, but there might not be noise
  if 0 not in categories:
    numCategories += 1
  repetitionCounter = np.zeros((numCategories,))
  lastCategory = None
  repetition = []
  tmActiveCellsClusters = {i: [] for i in range(numCategories)}
  tmPredictedActiveCellsClusters = {i: [] for i in range(numCategories)}
  tpActiveCellsClusters = {i: [] for i in range(numCategories)}
  for i in range(len(traces['actualCategory'])):
    category = int(traces['actualCategory'][i])
    tmPredictedActiveCells = traces['tmPredictedActiveCells'][i]
    tmActiveCells = traces['tmActiveCells'][i]
    tpActiveCells = traces['tpActiveCells'][i]

    if category != lastCategory:
      repetitionCounter[category] += 1
    lastCategory = category
    repetition.append(repetitionCounter[category] - 1)

    tmActiveCellsClusters[category].append(tmActiveCells)
    tmPredictedActiveCellsClusters[category].append(tmPredictedActiveCells)
    tpActiveCellsClusters[category].append(tpActiveCells)

  assert len(traces['actualCategory']) == sum([len(tpActiveCellsClusters[i])
                                               for i in range(numCategories)])

  return {
    'tmActiveCells': tmActiveCellsClusters,
    'tmPredictedActiveCells': tmPredictedActiveCellsClusters,
    'tpActiveCells': tpActiveCellsClusters,
    'repetition': repetition,
  }



def vizInterCategoryClusters(traces, outputDir, cellsType,
                             numCells, pointsToPlot=100):
  sdrs = convertNonZeroToSDR(traces[cellsType][-pointsToPlot:], numCells)

  clusterAssignments = traces['actualCategory'][-pointsToPlot:]
  numClasses = len(set(clusterAssignments))

  npos, distanceMat = project2D(sdrs)

  title = 'Actual category clusters in 2D (using %s)' % cellsType
  outputFile = '%s/%s' % (outputDir, title)
  viz2DProjection(title, outputFile, numClasses, clusterAssignments, npos)
  title = 'Actual category clusters distances (using %s)' % cellsType
  outputFile = '%s/%s' % (outputDir, title)
  plotDistanceMat(distanceMat, title, outputFile)



def vizInterSequenceClusters(traces, outputDir, cellsType, numCells, 
                             numClasses, ignoreNoise=True):
  clusters = assignClusters(traces)

  # compare inter-cluster distance over time
  numRptsPerCategory = {}
  categories = np.unique(traces['actualCategory'])
  repetition = np.array(clusters['repetition'])
  for category in categories:
    numRptsPerCategory[category] = np.max(
      repetition[np.array(traces['actualCategory']) == category])

  SDRclusters = []
  clusterAssignments = []
  numRptsMin = np.min(numRptsPerCategory.values()).astype('int32')
  for rpt in range(numRptsMin + 1):
    idx0 = np.logical_and(np.array(traces['actualCategory']) == 0,
                          repetition == rpt)
    idx1 = np.logical_and(np.array(traces['actualCategory']) == 1,
                          repetition == rpt)
    idx2 = np.logical_and(np.array(traces['actualCategory']) == 2,
                          repetition == rpt)

    c0slice = [traces['tmPredictedActiveCells'][i] for i in range(len(idx0)) if
               idx0[i]]
    c1slice = [traces['tmPredictedActiveCells'][i] for i in range(len(idx1)) if
               idx1[i]]
    c2slice = [traces['tmPredictedActiveCells'][i] for i in range(len(idx2)) if
               idx2[i]]

    if not ignoreNoise:
      SDRclusters.append(c0slice)
      clusterAssignments.append(0)
    SDRclusters.append(c1slice)
    clusterAssignments.append(1)
    SDRclusters.append(c2slice)
    clusterAssignments.append(2)

    print " Presentation #{}: ".format(rpt)
    if not ignoreNoise:
      d01 = clusterDist(c0slice, c1slice, numCells)
      print '=> d(c0, c1): %s' % d01
      d02 = clusterDist(c0slice, c2slice, numCells)
      print '=> d(c0, c2): %s' % d02

    d12 = clusterDist(c1slice, c2slice, numCells)
    print '=> d(c1, c2): %s' % d12

  npos, distanceMat = projectClusters2D(SDRclusters, numCells)
  title = 'Inter-sequence clusters in 2D (using %s)' % cellsType
  outputFile = '%s/%s' % (outputDir, title)
  viz2DProjection(title, outputFile, numClasses, clusterAssignments, npos)
  title = 'Inter-sequence clusters distances (using %s)' % cellsType
  outputFile = '%s/%s' % (outputDir, title)
  plotDistanceMat(distanceMat, title, outputFile)
