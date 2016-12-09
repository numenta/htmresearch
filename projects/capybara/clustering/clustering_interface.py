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


  @abstractmethod
  def infer(self, point):
    """
    Find the best cluster for an input point.
    
    :param point: (Point) input point.
    :return confidence, closest: (Cluster) best cluster for the input point.
    """
    raise NotImplementedError()


  @abstractmethod
  def learn(self, new_cluster):
    """
    Learn a new cluster. Once the cluster is learned, it can be used in the
    inference step to make predictions. 
    
    :param new_cluster: (Cluster) cluster of points to learn.
    """
    raise NotImplementedError()


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


  def prune(self, max_num_clusters):
    """
    If there are more clusters clusters than the max number of clusters 
    allowed, merge the closest clusters.
    
    :param max_num_clusters: (int) max number of clusters allowed.
    """
    while len(self.clusters) >= max_num_clusters:
      self.merge_closest_clusters()


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
