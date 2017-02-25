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


from htmresearch.frameworks.clustering.dim_reduction import (project2D,
                                                             assignClusters,
                                                             viz2DProjection,
                                                             plotDistanceMat)
from htmresearch.frameworks.clustering.utils import generateSDRs



def main():
  numClasses = 7
  numSDRsPerClass = 20
  noiseLevel = 0.1
  vizTitle = '2D projection, noise level: {}'.format(noiseLevel)

  # SDR parameters
  n = 1024
  w = 20

  sdrs = generateSDRs(numClasses, numSDRsPerClass, n, w, noiseLevel)

  clusterAssignments = assignClusters(sdrs, numClasses, numSDRsPerClass)

  npos, distanceMat = project2D(sdrs, method='mds')

  outputFile = '2d_projections.png'
  viz2DProjection(vizTitle, outputFile, numClasses, clusterAssignments, npos)
  outputFile = 'distance_matrix.png'
  plotDistanceMat(distanceMat, 'Inter-cluster distances', outputFile,
                  showPlot=True)



if __name__ == '__main__':
  main()
