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

import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy
import scipy.sparse

from nupic.algorithms.KNNClassifier import KNNClassifier



class HierarchicalClustering(object):
  """
  Implements hierarchical agglomerative clustering on the output of a classification network.

  The dissimilarity measure used is the negative overlap between SDRs.
  """


  def __init__(self, knn):
    """
    Initialization for HierarchicalClustering object.
    
    @param knn (nupic.algorithms.KNNClassifier) Populated instance of KNN
        classifer from which to draw training vectors.
    """
    self._knn = knn
    self._overlaps = None


  def cluster(self, linkageMethod="single"):
    """
    Perform hierarchical clustering on training vectors using specified linkage
    method. Results can be obtained using getLinkageMatrix(), etc.
    
    @param linkageMethod (string) Linkage method for computing between-class
        dissimilarities. Valid options are: "single" (aka min), "complete"
        (aka max), "average", and "weighted". For more information, see
        http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    if self._overlaps is None:
      self._populateOverlaps()
    overlaps = self._overlaps

    linkage = scipy.cluster.hierarchy.linkage(-overlaps, method=linkageMethod)
    self._linkage = linkage


  def getLinkageMatrix(self):
    """
    Returns a linkage matrix of the form defined by
    http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    return self._linkage.copy()


  def getDendrogram(self, truncate_mode=None, p=30):
    linkage = self.getLinkageMatrix()
    linkage[:,2] -= numpy.min(linkage[:,2])

    fig = plt.figure()
    ax = plt.axes()
    scipy.cluster.hierarchy.dendrogram(linkage,
      p=p, truncate_mode=truncate_mode, ax=ax)
    return fig


  ##################
  # Helper Methods #
  ##################


  def _populateOverlaps(self):
    sparseDataMatrix = HierarchicalClustering._extractVectorsFromKNN(self._knn)
    self._overlaps = HierarchicalClustering._computeOverlaps(sparseDataMatrix)


  @staticmethod
  def _extractVectorsFromKNN(knn):
    dim = len(knn.getPattern(0, sparseBinaryForm=False))
    sparseRowList = []

    for i in xrange(knn._numPatterns):
      nzIndices = knn.getPattern(i, sparseBinaryForm=True)
      sparseRow = scipy.sparse.csr_matrix(
        (numpy.ones(len(nzIndices), dtype=bool),
        (numpy.zeros(len(nzIndices)), nzIndices)),
        shape=(1,dim))
      sparseRowList.append(sparseRow)
    
    sparseDataMatrix = scipy.sparse.vstack(sparseRowList)

    return sparseDataMatrix


  @staticmethod
  def _computeOverlaps(data, selfOverlaps=False, dtype="int16"):
    """
    Calculates all pairwise overlaps between the rows of the input. Returns an
    array of all n(n-1)/2 values in the upper triangular portion of the
    pairwise overlap matrix. Values are returned in row-major order.

    @param data (scipy.sparse.csr_matrix) A CSR sparse matrix with one vector
        per row. Any non-zero value is considered an active bit.

    @param selfOverlaps (boolean) If true, include diagonal (density) values
        from the pairwise similarity matrix. Then the returned vector has
        n(n+1)/2 elements. Optional, defaults to False.
    
    @param dtype (string) Data type of returned array in numpy dtype format.
        Optional, defaults to 'int16'.
    
    @returns (numpy.ndarray) A vector of pairwise overlaps as described above.
    """
    nVectors = data.shape[0]
    nDims = data.shape[1]
    nPairs = (nVectors+1)*nVectors/2 if selfOverlaps else (
      nVectors*(nVectors-1)/2)
    overlaps = numpy.ndarray(nPairs, dtype=dtype)
    pos = 0

    for i in xrange(nVectors):
      start = i if selfOverlaps else i+1
      a = data[i]
      b = data[start:]
      newOverlaps = a.multiply(b).getnnz(1)
      run = newOverlaps.shape[0]
      overlaps[pos:pos+run] = newOverlaps
      pos += run
    return overlaps

