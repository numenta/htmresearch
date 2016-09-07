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
import numpy as np
from optparse import OptionParser
from itertools import permutations

from htmresearch.frameworks.classification.utils.traces import loadTraces
from htmresearch.frameworks.clustering.dim_reduction import (project2D,
                                                             viz2DProjection,
                                                             plotDistanceMat)
from htmresearch.frameworks.clustering.distances import percentOverlap



def _getArgs():
  parser = OptionParser(usage="Analyze SDR clusters")

  parser.add_option("-f",
                    "--fileName",
                    type=str,
                    default='results/traces_binary_sp-True_tm-True_'
                            'tp-False_KNNClassifier.csv',
                    dest="fileName",
                    help="fileName of the csv trace file")

  (options, remainder) = parser.parse_args()
  return options, remainder



def convertNonZeroToSDR(patternNZs):
  sdrs = []
  for patternNZ in patternNZs:
    sdr = np.zeros(numCells)
    sdr[patternNZ] = 1
    sdrs.append(sdr)

  return sdrs



def vizCellStates(traces, cellsType, numCells, startFrom=0):
  sdrs = convertNonZeroToSDR(traces['tmActiveCells'][startFrom:])

  clusterAssignments = traces['actualCategory'][startFrom:]
  numClasses = len(set(clusterAssignments))

  npos, distanceMat = project2D(sdrs)

  vizTitle = 'TM active cells projections'
  viz2DProjection(vizTitle, numClasses, clusterAssignments, npos)

  plotDistanceMat(distanceMat)



def assignClusters(traces):
  tmActiveCellsClusters = {}
  tmPredictedActiveCellsClusters = {}
  tpActiveCellsClusters = {}
  numCategories = len(np.unique(traces['actualCategory']))
  repetitionCounter = np.zeros((numCategories,))
  lastCategory = None
  repetition = []
  for i in range(len(traces['actualCategory'])):
    category = int(traces['actualCategory'][i])
    tmPredictedActiveCells = traces['tmPredictedActiveCells'][i]
    tmActiveCells = traces['tmActiveCells'][i]
    tpActiveCells = traces['tpActiveCells'][i]

    if category != lastCategory:
      repetitionCounter[category] += 1
    lastCategory = category
    repetition.append(repetitionCounter[category] - 1)
    if category not in tmActiveCellsClusters:
      tmActiveCellsClusters[category] = [tmActiveCells]
    else:
      tmActiveCellsClusters[category].append(tmActiveCells)

    if category not in tmPredictedActiveCellsClusters:
      tmPredictedActiveCellsClusters[category] = [tmPredictedActiveCells]
    else:
      tmPredictedActiveCellsClusters[category].append(tmPredictedActiveCells)

    if category not in tpActiveCellsClusters:
      tpActiveCellsClusters[category] = [tpActiveCells]
    else:
      tpActiveCellsClusters[category].append(tpActiveCells)

  assert len(traces['actualCategory']) == sum([len(tpActiveCellsClusters[i])
                                               for i in [0, 1, 2]])

  return {
    'tmActiveCells': tmActiveCellsClusters,
    'tmPredictedActiveCells': tmPredictedActiveCellsClusters,
    'tpActiveCells': tpActiveCellsClusters,
    'repetition': repetition,
  }



def clusterDist(c1, c2):
  """
  Distance between 2 clusters

  :param c1: (np.array) cluster 1
  :param c2: (np.array) cluster 2
  :return: distance between 2 clusters
  """
  minDists = []
  for sdr1 in c1:
    d = []
    for sdr2 in c2:
      d.append(1 - percentOverlap(sdr1, sdr2))
    minDists.append(min(d))

  return np.mean(minDists)


def meanInClusterDistances(cluster):
  overlaps = []
  perms = list(permutations(cluster, 2))
  for (sdr1, sdr2) in perms:
    overlap = percentOverlap(sdr1, sdr2)
    overlaps.append(overlap)
  return sum(overlaps) / len(overlaps)
  
  
  
if __name__ == "__main__":
  (_options, _args) = _getArgs()
  fileName = _options.fileName

  traces = loadTraces(fileName)
  cellsType = 'tmActiveCells'
  numCells = 1024 * 4
  numSteps = len(traces['step'])
  startFrom = int(numSteps * 0.6)
  vizCellStates(traces, cellsType, numCells, startFrom=100)

  clusters = assignClusters(traces)
  tmActiveCellsClusters = [convertNonZeroToSDR(clusters['tmActiveCells'][i])
                           for i in range(3)]
  
  c0 = tmActiveCellsClusters[0]
  c1 = tmActiveCellsClusters[1]
  c2 = tmActiveCellsClusters[2]
  
  print 'inter-cluster disatnces:'
  d01 = clusterDist(c0, c1)
  d02 = clusterDist(c0, c2)
  d12 = clusterDist(c1, c2)
  print '=> d(c0, c1): %s' %d01
  print '=> d(c0, c2): %s' %d02
  print '=> d(c1, c2): %s' %d12

  # compare c1 - c2 distance over time
  numRptsPerCategory = {}
  categories = np.unique(traces['actualCategory'])
  repetition = np.array(clusters['repetition'])
  for category in categories:
    numRptsPerCategory[category] = np.max(
      repetition[np.array(traces['actualCategory']) == category])

  numRptsMin = np.min(numRptsPerCategory.values()).astype('int32')
  for rpt in range(numRptsMin+1):
    idx0 = np.logical_and(np.array(traces['actualCategory']) == 0,
                          repetition == rpt)
    idx1 = np.logical_and(np.array(traces['actualCategory']) == 1,
                          repetition == rpt)
    idx2 = np.logical_and(np.array(traces['actualCategory']) == 2,
                          repetition == rpt)

    c0slice = [traces['tmActiveCells'][i] for i in range(len(idx0)) if idx0[i]]
    c1slice = [traces['tmActiveCells'][i] for i in range(len(idx1)) if idx1[i]]
    c2slice = [traces['tmActiveCells'][i] for i in range(len(idx2)) if idx2[i]]

    d01 = clusterDist(convertNonZeroToSDR(c0slice),
                      convertNonZeroToSDR(c1slice))
    d02 = clusterDist(convertNonZeroToSDR(c0slice),
                      convertNonZeroToSDR(c2slice))
    d12 = clusterDist(convertNonZeroToSDR(c1slice),
                      convertNonZeroToSDR(c2slice))

    print " Presentation # {} : ".format(rpt)
    print '=> d(c1, c2): %s' % d12

  print 'mean in-cluster distances:'
  print '=> c0 mean in-cluster dist: %s' % meanInClusterDistances(c0)
  print '=> c1 mean in-cluster dist: %s' % meanInClusterDistances(c1)
  print '=> c2 mean in-cluster dist: %s' % meanInClusterDistances(c2)


