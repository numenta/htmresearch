import itertools
import logging
import numpy as np

from htmresearch.frameworks.clustering.distances import (
  computeClusterDistances, overlapDistance)

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)



class Point(object):
  def __init__(self, pointId, value, label):
    self._id = pointId
    self._value = value
    self._label = label


  def getId(self):
    return self._id


  def setLabel(self, label):
    self._label = label


  def getLabel(self):
    return self._label


  def setValue(self, value):
    self._value = value


  def getValue(self):
    return self._value



class Cluster(object):
  def __init__(self, id, timeCreated, points=None, label=None):
    self._id = id
    self._label = label
    if points:
      self._points = points
    else:
      self._points = []
    _LOGGER.debug('CREATE: New cluster %s created!' % self._id)

    self._timeCreated = timeCreated
    self._lastUpdated = timeCreated
    self._numPointsClustered = 0


  def __repr__(self):
    return repr({
      '_id': self._id, '_label': self._label, 'numPoints': len(self._points),
      '_timeCreated': self._timeCreated, '_lastUpdated': self._lastUpdated,
      '_numPointsClustered': self._numPointsClustered
    })


  def add(self, point, updateTime):
    self._points.append(point)
    self._lastUpdated = updateTime
    self._numPointsClustered += 1


  def setLabel(self, name):
    self._label = name


  def getLabel(self):
    return self._label


  def getId(self):
    return self._id


  def setPoints(self, points):
    self._points = points


  def getPoints(self):
    return self._points


  def size(self):
    return len(self._points)


  def merge(self, cluster, updateTime):
    for point in cluster.getPoints():
      self.add(point, updateTime)



class Clustering(object):
  def __init__(self,
               mergeThreshold,
               anomalousThreshold,
               stableThreshold,
               minClusterSize,
               pointSimilarityThreshold,
               pruningFrequency=None,
               prune=False,
               fistClusterId=0):

    # Clusters
    self._numIterations = 0
    self._newCluster = Cluster(fistClusterId, self._numIterations)
    self._clusters = {}

    # Anomaly Score Thresholds
    self._anomalousThreshold = anomalousThreshold  # to create new cluster
    self._stableThreshold = stableThreshold  # to add point to cluster

    # Cluster distance threshold
    self._mergeThreshold = mergeThreshold
    self._minClusterSize = minClusterSize
    self._similarityThreshold = pointSimilarityThreshold

    # Cluster pruning
    self._prune = prune
    self._pruningFrequency = pruningFrequency
    self._clusterIdCounter = fistClusterId + 1


  def getClusterById(self, clusterId):
    return self._clusters[clusterId]


  def getClusters(self):
    return self._clusters.values()


  def getNewCluster(self):
    return self._newCluster


  def _addCluster(self, cluster):
    clusterId = cluster.getId()
    if clusterId not in self._clusters:
      self._clusters[clusterId] = cluster
    else:
      raise ValueError('Cluster ID %s is already in use.' % clusterId)
    _LOGGER.debug("Cluster %s permanently added" % clusterId)


  def _removeCluster(self, cluster):
    clusterId = cluster.getId()
    if clusterId in self._clusters:
      self._clusters.pop(clusterId)
      _LOGGER.debug('REMOVED: cluster with ID %s was removed' % clusterId)


  def _mergeNewCluster(self):

    clusterDistPairs = computeClusterDistances(self._newCluster,
                                               self.getClusters())
    clusterMerged = False
    if len(clusterDistPairs) > 0:
      closestClusterDist, closestCluster = clusterDistPairs[0]
      if closestClusterDist < self._mergeThreshold:
        _LOGGER.debug("MERGE: Cluster %s merged with cluster %s. "
                      "Inter-cluster distance: %s"
                      % (self._newCluster.getId(),
                         closestCluster.getId(),
                         closestClusterDist))
        updateTime = self._numIterations
        closestCluster.merge(self._newCluster, updateTime)
        clusterMerged = True

    return clusterMerged


  def _mergeCluster(self, cluster, clusters):
    """
    Merge cluster if it is close enough to a cluster in the list and return 
    True. Otherwise, don't merge the cluster and return False.
    :param cluster: (Cluster) cluster to merge w/ one of the existing clusters. 
    :param clusters: (list of Clusters) list of clusters to compare the first 
    cluster to.
    :return: (bool) Wether or not the cluster was merged.
    """

    clusterDistPairs = computeClusterDistances(cluster, clusters)
    clusterMerged = False
    if len(clusterDistPairs) > 0:
      for clusterDistPair in clusterDistPairs:
        closestClusterDist, closestCluster = clusterDistPair

        if closestClusterDist < self._mergeThreshold:
          _LOGGER.debug("MERGE: Cluster %s merged with cluster %s. "
                        "Inter-cluster distance: %s" % (cluster.getId(),
                                                        closestCluster.getId(),
                                                        closestClusterDist))
          closestCluster.merge(cluster, self._numIterations)
          self._removeCluster(cluster)
          clusterMerged = True

    return clusterMerged


  def _pruneClusters(self):
    # TODO: not needed for now. If needed later, make sure to go though all 
    # cluster permutations.
    raise NotImplementedError("Cluster pruning not implemented.")


  def infer(self):
    """
    Inference: find the closest cluster to the new cluster.
    """
    clusterDistPairs = computeClusterDistances(self._newCluster,
                                               self.getClusters())
    if len(clusterDistPairs) > 0:
      distToCluster, predictedCluster = clusterDistPairs[0]
      # Confidence of inference
      meanClusterDist = np.mean([p[0] for p in clusterDistPairs])
      if meanClusterDist > 0:
        confidence = 1 - (distToCluster / meanClusterDist)
      else:
        if len(self._clusters) > 1:
          raise ValueError("The mean distance can't be 0. Number of "
                           "clusters: %s" % len(self._clusters))
        else:
          confidence = 1
    else:
      predictedCluster = None
      confidence = -1

    return predictedCluster, confidence


  def cluster(self, sdrId, sdrValue, anomalyScore, trueLabel=None):

    point = Point(sdrId, sdrValue, trueLabel)

    # The data is anomalous
    if self._anomalousThreshold <= anomalyScore:
      if self._newCluster.size() >= self._minClusterSize:
        clusterMerged = self._mergeNewCluster()
        if not clusterMerged:
          self._addCluster(self._newCluster)
      else:
        _LOGGER.debug('DELETE: Cluster %s discarded. Not enough points (%s '
              'points. Min cluster size is %s)' %
              (self._newCluster.getId(), 
               self._newCluster.size(),
               self._minClusterSize))
      

      self._newCluster = Cluster(self._clusterIdCounter,
                                 self._numIterations)
      self._clusterIdCounter += 1

      predictedCluster = None
      confidence = -3

    # The data is unstable, so do nothing
    elif self._stableThreshold <= anomalyScore:
      predictedCluster = None
      confidence = -2

    # The data is stable
    else:
      self._addPoint(point)
      if self._prune and self._numIterations % self._pruningFrequency == 0:
        self._pruneClusters()
      predictedCluster, confidence = self.infer()

    self._numIterations += 1
    return predictedCluster, confidence


  def _addPoint(self, point):

    if self._newCluster.size() >= 1:
      dists = []
      for p in self._newCluster.getPoints():
        d = overlapDistance(p.getValue(), point.getValue())
        dists.append(d)
      if min(dists) > self._similarityThreshold:
        self._newCluster.add(point, self._numIterations)
    else:
      self._newCluster.add(point, self._numIterations)


  def clusterActualCategoriesFrequencies(self):

    clusterActualCategoriesFrequencies = []
    for cluster in self._clusters.values():
      labels = []
      for point in cluster.getPoints():
        labels.append(int(point.getLabel()))
      unique, counts = np.unique(labels, return_counts=True)
      frequencies = []
      for actualCategory, numberOfPoints in np.asarray((unique, counts)).T:
        frequencies.append({
          'actualCategory': actualCategory,
          'numberOfPoints': numberOfPoints
        })
      clusterActualCategoriesFrequencies.append({
        'clusterId': cluster.getId(),
        'actualCategoryFrequencies': frequencies
      })

    return clusterActualCategoriesFrequencies
