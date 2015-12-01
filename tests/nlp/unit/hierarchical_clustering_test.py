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


  def testCondensedIndex(self):
    flat = range(6)

    # first try only indexing upper triangular region
    indicesA = [0, 0, 0, 1, 1, 2]
    indicesB = [1, 2, 3, 2, 3, 3]
    res = HierarchicalClustering._condensedIndex(indicesA, indicesB, 4)
    self.assertEqual(res.tolist(), flat)

    # ensure we get same result by transposing some indices for the lower
    # triangular region
    indicesA = [0, 2, 3, 1, 3, 2]
    indicesB = [1, 0, 0, 2, 1, 3]
    res = HierarchicalClustering._condensedIndex(indicesA, indicesB, 4)
    self.assertEqual(res.tolist(), flat)

    # finally check that we get an assertion error if we try accessing
    # an element from the diagonal
    with self.assertRaises(AssertionError):
      indicesA = [0, 2, 0, 1, 3, 2]
      indicesB = [1, 2, 3, 2, 1, 3]
      _ = HierarchicalClustering._condensedIndex(indicesA, indicesB, 4)


  def testGetPrototypes(self):
    data = scipy.sparse.csr_matrix([
      [1, 1, 0, 1],
      [1, 0, 1, 1],
      [0, 1, 1, 0],
      [1, 1, 1, 1]
    ])
    overlaps = HierarchicalClustering._computeOverlaps(data)

    prototypes = HierarchicalClustering._getPrototypes([0, 1, 2, 3], overlaps)
    self.assertEqual(set(prototypes.tolist()), set([3]))

    prototypes = HierarchicalClustering._getPrototypes([1, 2, 3], overlaps, 2)
    self.assertEqual(set(prototypes.tolist()), set([3, 1]))

    prototypes = HierarchicalClustering._getPrototypes([0, 2, 3], overlaps, 2)
    self.assertEqual(set(prototypes.tolist()), set([3, 0]))

    prototypes = HierarchicalClustering._getPrototypes([0, 1, 2], overlaps, 2)
    self.assertEqual(set(prototypes.tolist()), set([0, 1]))



if __name__ == "__main__":
  unittest.main()

