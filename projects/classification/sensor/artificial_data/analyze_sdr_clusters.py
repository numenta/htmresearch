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
import matplotlib.pyplot as plt
from sklearn import manifold
from optparse import OptionParser
from itertools import permutations

from htmresearch.frameworks.classification.utils.traces import loadTraces
from htmresearch.frameworks.clustering.dim_reduction import (project2D,
                                                             projectClusters2D,
                                                             viz2DProjection,
                                                             plotDistanceMat)
from htmresearch.frameworks.clustering.distances import (percentOverlap,
                                                         clusterDist)



def _getArgs():
  parser = OptionParser(usage="Analyze SDR clusters")

  parser.add_option("-f",
                    "--fileName",
                    type=str,
                    default='results/traces_sensortag_x_sp=True_tm=True_tp'
                            '=False_KNNClassifier.csv',
                    # default='results/traces_binary_ampl=10.0_mean=0.0_noise=0'
                    #         '.0_sp=True_tm=True_tp=False_KNNClassifier.csv',
                    dest="fileName",
                    help="fileName of the csv trace file")

  parser.add_option("--includeNoiseCategory",
                    type=str,
                    default=0,
                    dest="includeNoiseCategory",
                    help="whether to include noise category for viz")

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
                                               for i in [0, 1, 2]])

  return {
    'tmActiveCells': tmActiveCellsClusters,
    'tmPredictedActiveCells': tmPredictedActiveCellsClusters,
    'tpActiveCells': tpActiveCellsClusters,
    'repetition': repetition,
  }



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
  includeNoiseCategory = _options.includeNoiseCategory

  traces = loadTraces(fileName)
  cellsType = 'tmActiveCells'
  numCells = 1024 * 4
  numSteps = len(traces['recordNumber'])
  startFrom = int(numSteps * 0.6)

  # no clustering with individual cell states, remove?
  # vizCellStates(traces, cellsType, numCells, startFrom=100)

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

    if includeNoiseCategory:
      SDRclusters.append(c0slice)
      clusterAssignments.append(0)
    SDRclusters.append(c1slice)
    clusterAssignments.append(1)
    SDRclusters.append(c2slice)
    clusterAssignments.append(2)

    print " Presentation #{}: ".format(rpt)
    if includeNoiseCategory:
      d01 = clusterDist(convertNonZeroToSDR(c0slice),
                        convertNonZeroToSDR(c1slice))
      print '=> d(c0, c1): %s' % d01
      d02 = clusterDist(convertNonZeroToSDR(c0slice),
                        convertNonZeroToSDR(c2slice))
      print '=> d(c0, c2): %s' % d02
    
    d12 = clusterDist(convertNonZeroToSDR(c1slice),
                      convertNonZeroToSDR(c2slice))
    print '=> d(c1, c2): %s' % d12

  print " visualizing clusters with MDS "
  npos, distanceMat = projectClusters2D(SDRclusters)
  plt.figure()
  plt.imshow(distanceMat, interpolation="nearest")
  plt.colorbar()
  plt.xlabel('sequence #')
  plt.ylabel('sequence #')
  plt.savefig('results/cluster_distance_matrix_example.pdf')

  viz2DProjection('sequenceCluster', 3, clusterAssignments, npos)
