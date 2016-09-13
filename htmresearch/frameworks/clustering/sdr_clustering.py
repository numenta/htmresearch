import logging
import numpy as np

from htmresearch.frameworks.clustering.distances import clusterDist

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
  def __init__(self, id, points=None, label=None):
    self._id = id
    self._label = label
    if points:
      self._points = points
    else:
      self._points = []
    _LOGGER.debug("New cluster %s created!" % self._id)


  def add(self, point):
    self._points.append(point)


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


  def merge(self, cluster):
    for point in cluster.getPoints():
      self.add(point)



class Clustering(object):
  def __init__(self,
               mergeThreshold,
               anomalousThreshold,
               stableThreshold,
               fistClusterId=0):

    # Clusters
    self.newCluster = Cluster(fistClusterId)
    self.clusters = []

    # Anomaly Score Thresholds
    self.anomalousThreshold = anomalousThreshold  # to create new cluster
    self.stableThreshold = stableThreshold  # to add point to cluster

    # Cluster distance threshold
    self.mergeThreshold = mergeThreshold


  def findClosestClusters(self, cluster):
    """
    
    :param cluster: 
    :return: ordered list of clusters and their distances 
    """
    dists = []
    for c in self.clusters:
      d = clusterDist(c.getPoints(), cluster.getPoints())
      dists.append((d, c))

    return sorted(dists)


  def addCluster(self, cluster):
    self.clusters.append(cluster)
    _LOGGER.debug("Cluster %s permanently added" % cluster.getId())


  def addOrMergeCluster(self, cluster):

    clusterDistPairs = self.findClosestClusters(cluster)
    if len(clusterDistPairs) > 0:
      notMerged = True
      for clusterDistPair in clusterDistPairs: 
        closestClusterToNewDist, closestClusterToNew = clusterDistPair
  
        if closestClusterToNewDist < self.mergeThreshold:
          _LOGGER.debug("Cluster %s merged with cluster %s. Inter-cluster "
                        "distance: %s" % (cluster.getId(),
                                          closestClusterToNew.getId(),
                                          closestClusterToNewDist))
          closestClusterToNew.merge(cluster)
          notMerged = False
      if notMerged:
        self.addCluster(cluster)
    else:
      self.addCluster(cluster)


  def infer(self):
    """
    Inference: find the closest cluster to the new cluster.
    """
    clusterDistPairs = self.findClosestClusters(self.newCluster)
    if len(clusterDistPairs) > 0:
      distToCluster, predictedCluster = clusterDistPairs[0]
      # Confidence of inference
      meanClusterDist = np.mean([p[0] for p in clusterDistPairs])
      if meanClusterDist > 0:
        confidence = 1 - (distToCluster / meanClusterDist)
      else:
        if len(self.clusters) > 1:
          raise ValueError("The mean distance can't be 0. Number of "
                           "clusters: %s" % len(self.clusters))
        else:
          confidence = 1
    else:
      predictedCluster = None
      confidence = -1

    return predictedCluster, confidence


  def cluster(self, sdrId, sdrValue, anomalyScore, trueLabel=None):

    point = Point(sdrId, sdrValue, trueLabel)

    # The data is anomalous
    if self.anomalousThreshold <= anomalyScore:
      if self.newCluster.size() > 0:
        self.addOrMergeCluster(self.newCluster)
        self.newCluster = Cluster(len(self.clusters))

      predictedCluster = None
      confidence = -3

    # If the data is unstable, so do nothing
    elif self.stableThreshold <= anomalyScore < self.anomalousThreshold:
      predictedCluster = None
      confidence = -2

    # The data is stable
    else:
      self.newCluster.add(point)

      predictedCluster, confidence = self.infer()

    return predictedCluster, confidence


  def inClusterActualCategoriesFrequencies(self):

    inClusterActualCategoriesFrequencies = []
    for cluster in self.clusters:
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
      inClusterActualCategoriesFrequencies.append({
        'clusterId': cluster.getId(),
        'actualCategoryFrequencies': frequencies
      })

    return inClusterActualCategoriesFrequencies
