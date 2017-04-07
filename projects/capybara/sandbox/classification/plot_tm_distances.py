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
import os

from optparse import OptionParser
from itertools import permutations

from htmresearch.frameworks.classification.utils.traces import loadTraces
from htmresearch.frameworks.clustering.viz import (vizInterSequenceClusters,
                                                   vizInterCategoryClusters)
from htmresearch.frameworks.clustering.distances import (percentOverlap,
                                                         clusterDist)


def _getArgs():
  parser = OptionParser(usage="Analyze SDR clusters")

  parser.add_option("-f",
                    "--fileName",
                    type=str,
                    dest="fileName",
                    help="fileName of the csv trace file")

  parser.add_option("--includeNoiseCategory",
                    type=str,
                    default=0,
                    dest="includeNoiseCategory",
                    help="whether to include noise category for viz")

  (options, remainder) = parser.parse_args()
  return options, remainder



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
  outputDir = fileName[:-4]
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)
  cellsType = 'tmPredictedActiveCells'
  numCells = 2048 * 32
  numSteps = len(traces['recordNumber'])
  pointsToPlot = numSteps / 10
  numClasses = len(set(traces['actualCategory']))
  vizInterCategoryClusters(traces,
                           outputDir,
                           cellsType,
                           numCells,
                           pointsToPlot)

  vizInterSequenceClusters(traces, outputDir, cellsType, numCells,
                           numClasses)
