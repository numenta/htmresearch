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



class LinkageNotComputedException(Exception):
  pass


class HierarchicalClustering(object):
  """
  Implements hierarchical agglomerative clustering on the output of a
  classification network.

  The dissimilarity measure used is the negative overlap between SDRs.
  
  There are 3 steps that must be performed to use the class.
  1) The class must be initialized with a KNNClassifier instance, from which it
  extracts training vectors.
  2) The `cluster()` method must be called with a string parameter specifying
  the linkage function for agglomerative clustering. See docstring on `cluster`
  for more information. This method can take significant time to execute.
  3) Once `cluster()` is called, the visualization methods can be called.
  Currently supported visualization methods are `getDendrogram()` and
  `getClusterPrototypes()`.

  Note that steps 2 and 3 above can be repeated to visualize the same data
  clustered using different linkage functions.


  Example
  =======

  hc = HierarchicalClustering(knn)
  hc.cluster("complete")
  prototypes = hc.getClusterPrototypes(20, 5)
  """


  def __init__(self, knn):
    """
    Initialization for HierarchicalClustering object.
    
    @param knn (nupic.algorithms.KNNClassifier) Populated instance of KNN
        classifer from which to draw training vectors.
    """
    self._knn = knn
    self._overlaps = None
    self._linkage = None


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
    if self._linkage is None:
      raise LinkageNotComputedException
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


  def getClusterPrototypes(self, numClusters, numPrototypes=1):
    """
    Create numClusters flat clusters and find approximately numPrototypes
    prototypes per flat cluster. Returns an array with each row containing the
    indices of the prototypes for a single flat cluster.

    @param numClusters (int) Number of flat clusters to return (approximate).

    @param numPrototypes (int) Number of prototypes to return per cluster.

    @returns (numpy.ndarray) Array with rows containing the indices of the
        prototypes for a single flat cluster.
    """
    linkage = self.getLinkageMatrix()
    linkage[:, 2] -= linkage[:, 2].min()

    clusters = scipy.cluster.hierarchy.fcluster(
      linkage, numClusters, criterion="maxclust")
    prototypes = []

    for cluster_id in numpy.unique(clusters):
      ids = numpy.arange(len(clusters))[clusters == cluster_id]
      cluster_prototypes = HierarchicalClustering._getPrototypes(
        ids, self._overlaps, numPrototypes)
      prototypes.append(cluster_prototypes)

    return numpy.vstack(prototypes)


  ##################
  # Helper Methods #
  ##################


  @staticmethod
  def _getPrototypes(indices, overlaps, topNumber=1):
    """
    Given a compressed overlap array and a set of indices specifying a subset
    of those in that array, return the set of topNumber indices of vectors that
    have maximum average overlap with other vectors in `indices`.

    @param indices (arraylike) Array of indices for which to get prototypes.

    @param overlaps (numpy.ndarray) Condensed array of overlaps of the form
        returned by _computeOverlaps().

    @param topNumber (int) The number of prototypes to return. Optional,
        defaults to 1.

    @returns (numpy.ndarray) Array of indices of prototypes
    """
    # find the number of data points based on the length of the overlap array
    # solves for n: len(overlaps) = n(n-1)/2
    n = numpy.roots([1, -1, -2 * len(overlaps)]).max()
    k = len(indices)

    indices = numpy.array(indices, dtype=int)
    rowIdxs = numpy.ndarray((k, k-1), dtype=int)
    colIdxs = numpy.ndarray((k, k-1), dtype=int)

    for i in xrange(k):
      rowIdxs[i, :] = indices[i]
      colIdxs[i, :i] = indices[:i]
      colIdxs[i, i:] = indices[i+1:]

    idx = HierarchicalClustering._condensedIndex(rowIdxs, colIdxs, n)
    subsampledOverlaps = overlaps[idx]

    meanSubsampledOverlaps = subsampledOverlaps.mean(1)
    biggestOverlapSubsetIdxs = numpy.argpartition(
      -meanSubsampledOverlaps, topNumber)[:topNumber]

    return indices[biggestOverlapSubsetIdxs]


  @staticmethod
  def _condensedIndex(indicesA, indicesB, n):
    """
    Given a set of n points for which pairwise overlaps are stored in a flat
    array X in the format returned by _computeOverlaps (upper triangular of the
    overlap matrix in row-major order), this function returns the indices in X
    that correspond to the overlaps for the pairs of points specified.

    Example
    -------
    Consider the case with n = 5 data points for which pairwise overlaps are
    stored in array X, which has length 10 = n(n-1)/2. To obtain the overlap
    of points 2 and 3 and the overlap of points 4 and 1, call 

      idx = _condensedIndex([2, 4], [3, 1], 5) # idx == [6, 1]

    Note: Since X does not contain the diagonal (self-comparisons), it is
    invalid to pass arrays such that indicesA[i] == indicesB[i] for any i.

    @param indicesA (arraylike) First dimension of pairs of datapoint indices

    @param indicesB (arraylike) Second dimension of pairs of datapoint indices

    @param n (int) Number of datapoints

    @returns (numpy.ndarray) Indices in condensed overlap matrix containing
        specified overlaps. Dimension will be same as indicesA and indicesB.
    """
    indicesA = numpy.array(indicesA, dtype=int)
    indicesB = numpy.array(indicesB, dtype=int)
    n = int(n)

    # Ensure that there are no self-comparisons
    assert (indicesA != indicesB).all()

    # re-arrange indices to ensure that rowIxs[i] < colIxs[i] for all i
    rowIxs = numpy.where(indicesA < indicesB, indicesA, indicesB)
    colIxs = numpy.where(indicesA < indicesB, indicesB, indicesA)

    flatIxs = rowIxs * (n - 1) - (rowIxs + 1) * rowIxs / 2 + colIxs - 1

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

