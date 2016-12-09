# ----------------------------------------------------------------------
#  Copyright (C) 2016, Numenta Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc. No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.
# ----------------------------------------------------------------------
import numpy as np

from abc import ABCMeta, abstractmethod
from queue import PriorityQueue



class Point(object):
  def __init__(self, value, label=None):
    """
    Point holding the value of an SDR and its optional label (ground truth).

    :param value: (np.Array) point value (SDR)
    :param label: (int) point label
    """
    self.value = value
    self.label = label



class Cluster(object):
  def __init__(self, id, center):
    """
    Cluster of points.
    
    :param id: (int) ID of the cluster
    :param center: (Point) center of the cluster
    """
    self.id = id
    self.center = center
    self.points = []
    self.size = 0


  def add(self, point):
    """
    Add point to cluster.
    
    :param point: (Point) point to add to cluster.
    """
    assert type(point) == Point
    self.points.append(point)
    if self.center is None:
      self.center = point
    self.center.value = ((self.center.value * self.size + point.value) /
                         float(self.size + 1))
    self.size += 1


  def merge(self, cluster):
    """
    Merge a cluster into this cluster.
    
    :param cluster: (Cluster) cluster to merge into this cluster. 
    """
    self.center.value = ((self.center.value * self.size +
                          cluster.center.value * cluster.size) /
                         float(self.size + cluster.size))
    self.size += cluster.size
    while len(cluster.points) > 0:
      point = cluster.points.pop()
      self.points.append(point)


  def label_distribution(self):
    """
    Returns distribution of each label in this cluster. E.g:
    [
      {
        'label': 1,
        'num_points': 20
      },
         ...
      {
        'label': 5,
        'num_points': 30
      }   
    ]
    """
    labels = [p.label for p in self.points]
    unique, counts = np.unique(labels, return_counts=True)
    label_distribution = []
    for label, num_points in np.asarray((unique, counts)).T:
      label_distribution.append({
        'label': label,
        'num_points': num_points
      })

    return label_distribution



class ClusteringInterface(object):
  __metaclass__ = ABCMeta


  def __init__(self, distance_func):
    """
    Clustering algorithm.
    
    :param distance_func: (function) distance metric. The function signature 
      is "distance_func(p1, p2)" where p1 and p2 are instances of Point.
    """
    self.distance_func = distance_func
    self.clusters = {}  # Keys are cluster IDs; Values are Clusters.


  def infer(self, point):
    """
    Find the closest cluster to a point.
    
    :param point: (Point) input point
    :return closest: (Cluster) closet cluster to the point.
    """

    return self.find_closest_cluster(point)


  def prune(self, max_num_clusters):
    """
    If there are too many clusters clusters than the max number of clusters 
    allowed, merge the closest clusters.
    
    :param max_num_clusters: (int) max number of clusters allowed.
    """
    while len(self.clusters) >= max_num_clusters:
      self.merge_closest_clusters()


  @staticmethod
  def noisy_sequence(anomaly_score, noisy_anomaly_score=0.3):
    """
    Determine whether a temporal sequence is noisy.
    
    :param anomaly_score: (float) anomaly score of the temporal memory
    :param noisy_anomaly_score: (float) threshold to determine whether the 
      anomaly score is noisy.
    :return: (bool) whether the sequence is noisy 
    """
    if anomaly_score > noisy_anomaly_score:
      return True
    else:
      return False


  @staticmethod
  def stable_sequence(anomaly_score, stable_anomaly_score=0.2):
    """
    Determine whether a temporal sequence is stable.
    
    :param anomaly_score: (float) anomaly score of the temporal memory
    :param stable_anomaly_score: (float) threshold to determine whether the 
      anomaly score is stable.
    :return: (bool) whether the sequence is stable 
    """
    if anomaly_score < stable_anomaly_score:
      return True
    else:
      return False


  def average_cluster_distance(self):
    """
    Average cluster distance between clusters.
    
    :return average_distance: (float) average distance between clusters. 
    """
    cluster_distances = []
    cluster_ids = self.clusters.keys()

    numClusters = len(cluster_ids)
    for i in range(numClusters):
      for j in range(i + 1, numClusters):
        ci = self.clusters[cluster_ids[i]].center.value
        cj = self.clusters[cluster_ids[j]].center.value
        d = self.distance_func(ci, cj)
        cluster_distances.append(d)

    if len(cluster_distances) > 0:
      return np.mean(cluster_distances)
    else:
      return 0.0


  def add_or_merge_cluster(self, cluster, merge_threshold):
    """
    Add cluster to the existing clusters or merge it with the closest cluster.
    
    :param cluster: (Cluster) the cluster to assign
    :param merge_threshold: (float) If the distance to the closest 
      cluster is below the merge threshold, the cluster will be merged with 
      the closest cluster. Otherwise, if the distance to the closest cluster 
      is  above the merge threshold, then the cluster will be added to the 
      existing clusters.
    """
    distance_to_closest, closest, = self.find_closest_cluster(cluster.center)
    if closest and distance_to_closest < merge_threshold:
      closest.merge(cluster)
    else:
      self.add_cluster(cluster)


  def create_cluster(self, center=None):
    """
    Create a cluster.
    
    :param center: (Point) optional center of the cluster. If a center is 
      provided, it will be added to the cluster list of points.
    """
    cluster_id = len(self.clusters) + 1
    cluster = Cluster(cluster_id, center)
    if center:
      cluster.add(center)
    return cluster


  def add_cluster(self, cluster):
    """
    Add cluster to the existing clusters.
    
    :param cluster: (Cluster) cluster to add.
    :raise: (ValueError) raise error if the cluster ID is already used. 
    """
    if cluster.id in self.clusters:
      raise ValueError('Cluster ID %s already exists' % cluster.id)
    self.clusters[cluster.id] = cluster


  @abstractmethod
  def find_closest_cluster(self, point):
    """
    Find the closest cluster to a point.
    
    :param point: (Point) The point of interest.
    :return distance_to_closest: (float) distance between closest cluster 
      center and point.
    :return closest: (Cluster) closest cluster to point.
    """
    raise NotImplementedError()


  def merge_closest_clusters(self):
    """
    Merge closest two clusters.
    """
    inter_cluster_dists = PriorityQueue()

    numClusters = len(self.clusters)
    for i in range(numClusters):
      for j in range(i + 1, numClusters):
        c1 = self.clusters.values()[i]
        c2 = self.clusters.values()[j]
        d = self.distance_func(c1.center.value, c2.center.value)
        inter_cluster_dists.put(InterClusterDist(c1, c2, d))

    smallest_inter_cluster_dist = inter_cluster_dists.get()
    cluster_to_merge = smallest_inter_cluster_dist.c2
    smallest_inter_cluster_dist.c1.merge(cluster_to_merge)
    del self.clusters[cluster_to_merge.id]



class InterClusterDist(object):
  """
  Inter-cluster distance.
   
  Useful to define cmp() for queue.PriorityQueue.
  """


  def __init__(self, c1, c2, c1_c2_dist):
    self.c1 = c1
    self.c2 = c2
    self.dist = c1_c2_dist


  def __cmp__(self, inter_cluster_dist):
    return cmp(self.dist, inter_cluster_dist.dist)


  def __str__(self):
    return "InterClusterDist(%f)" % self.dist
