#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy
import os
import scipy.sparse
import unittest

from mock import patch

from htmresearch.algorithms.hierarchical_clustering import HierarchicalClustering
from nupic.algorithms.KNNClassifier import KNNClassifier



class TestHierarchicalClustering(unittest.TestCase):


  def testComputeOverlapsWithoutDiagonal(self):
    data = scipy.sparse.csr_matrix([
      [1, 1, 0, 1],
      [0, 1, 1, 0],
      [1, 1, 1, 1]
    ])
    dists = HierarchicalClustering._computeOverlaps(data, selfOverlaps=False)
    self.assertEqual(dists.shape, (3,))
    self.assertEqual(dists.tolist(), [1, 3, 2])


  def testComputeOverlapsWithDiagonal(self):
    data = scipy.sparse.csr_matrix([
      [1, 1, 0, 1],
      [0, 1, 1, 0],
      [1, 1, 1, 1]
    ])
    dists = HierarchicalClustering._computeOverlaps(data, selfOverlaps=True)
    self.assertEqual(dists.shape, (6,))
    self.assertEqual(dists.tolist(), [3, 1, 3, 2, 2, 4])


  def testExtractVectorsFromKNN(self):
    vectors = numpy.random.rand(10, 25) < 0.1

    # Populate KNN
    knn = KNNClassifier()
    for i in xrange(vectors.shape[0]):
      knn.learn(vectors[i], 0)

    # Extract vectors from KNN
    sparseDataMatrix = HierarchicalClustering._extractVectorsFromKNN(knn)

    self.assertEqual(sparseDataMatrix.todense().tolist(), vectors.tolist())


if __name__ == "__main__":
  unittest.main()

