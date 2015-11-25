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
    n = linkage.shape[0]

    def llf(id):
      if id < n:
        return "leaf: " + str(id)
      else:
        return '[%d %d %1.2f]' % (id, 2, linkage[n-id,3])

    fig = plt.figure(figsize=(10,8), dpi=400)
    ax = plt.axes()
    scipy.cluster.hierarchy.dendrogram(linkage,
      p=p, truncate_mode=truncate_mode, ax=ax, orientation="right", leaf_font_size=5,
      no_labels=False, leaf_rotation=45.0, show_leaf_counts=True, leaf_label_func=llf,
      color_threshold=50)
    
    for label in ax.get_yticklabels():
      label.set_fontsize(4)

    return fig


  ##################
  # Helper Methods #
  ##################


  @staticmethod
  def getMaxAverageOverlap(indices, overlaps):
    # find the number of data points based on the length of the overlap array
    # solves for n: len(overlaps) = n(n-1)/2
    n = numpy.roots([1, -1, -2 * len(overlaps)]).max()
    k = len(indices)

    rowIdxs = numpy.ndarray((k, k-1), dtype=int)
    colIdxs = numpy.ndarray((k, k-1), dtype=int)

    for i in xrange(k):
      rowIdx[i, :] = indices[i]
      colIdx[i, :i] = indices[:i]
      colIdx[i, i:] = indices[i+1:]

    idx = HierarchicalClustering._condensedIndex(rowIdx, colIdx, n)
    subsampledOverlaps = overlaps[idx]
    biggestOverlapSubsetIdx = subsampledOverlaps.mean(1).argmax()
    return indices[biggestOverlapSubsetIdx]


  @staticmethod
  def _condensedIndex(indicesA, indicesB, n):
    """
    Given a set of n points for which pairwise overlaps are stored in a flat
    array X in the format returned by _computeOverlaps (upper triangular of the
    overlap matrix in row-major order), this function returns the indices in X
    for that correspond to the overlaps for pairs provided in the parameters.

    Example
    -------
    Consider the case with n = 5 data points for which pairwise overlaps are
    stored in array X, which has length 10 = n(n-1)/2. If we want the overlap
    of points 2 and 3 and the overlap of points 4 and 1, we would call 

      idx = _condensedIndex([2, 4], [3, 1], 5) # idx == [6, 1]

    Note: Since X does not contain the diagonal (self-comparisons), it is
    invalid to pass arrays such that indicesA[i] == indicesB[i] for any i.

    @param indicesA (arraylike) First dimension of pairs of datapoint indices

    @param indicesB (arraylike) Second dimension of pairs of datapoint indices

    @param n (int) Number of datapoints

    @returns (numpy.array) Indices in condensed overlap matrix containing
        specified overlaps. Dimension will be same as indicesA and indicesB.
    """
    # Ensure that there are no self-comparisons
    assert (indicesA != indicesB).all()

    # re-arrange indices to ensure that rowIxs[i] < colIxs[i] for all i
    rowIxs = numpy.where(indicesA < indicesB, indicesA, indicesB)
    colIxs = numpy.where(indicesA < indicesB, indicesB, indicesA)

    # compute the indices in X of the start of each row in the upper triangular
    flatRowStarts = rowIxs * n - (rowIxs + 1) * rowIxs / 2
    flatIxs = flatRowStarts + colIxs

    return flatIxs


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

