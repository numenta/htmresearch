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
from clustering_interface import ClusteringInterface



class OnlineClusteringV2(ClusteringInterface):
  def __init__(self, distance_func, merge_threshold=3.15):
    """
    Online clustering implementation. 
    
    :param distance_func: (function) distance metric. The function signature 
      is "distance_func(p1, p2)" where p1 and p2 are instances of Point.
    :param merge_threshold: (float) cutoff distance to add or merge new 
      clusters.      
    """
    super(OnlineClusteringV2, self).__init__(distance_func)
    self.merge_threshold = merge_threshold


  def infer(self, point):
    """
    Find the best cluster for an input point.
    
    :param point: (Point) input point.
    :return confidence, closest: (Cluster) best cluster for the input point.
    """
    (average_cluster_distance,
     distance_to_closest,
     closest) = self._find_closest_cluster(point)
    if closest:
      confidence = 1 - (distance_to_closest / average_cluster_distance)
    else:
      confidence = None
    return confidence, closest


  def learn(self, new_cluster):
    """
    Learn a new cluster. Once the cluster is learned, it can be used in 
    inference step to make predictions. 
    
    :param new_cluster: (Cluster) cluster of points to learn.
    """
    if self.merge_threshold is None:
      merge_threshold = self._average_cluster_distance() / 2.0
    else:
      merge_threshold = self.merge_threshold
    self._add_or_merge_cluster(new_cluster, merge_threshold)


  def _average_cluster_distance(self):
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


  def _add_or_merge_cluster(self, cluster, merge_threshold):
    """
    Add cluster to the existing clusters or merge it with the closest cluster.
    
    :param cluster: (Cluster) the cluster to assign
    :param merge_threshold: (float) If the distance to the closest 
      cluster is below the merge threshold, the cluster will be merged with 
      the closest cluster. Otherwise, if the distance to the closest cluster 
      is  above the merge threshold, then the cluster will be added to the 
      existing clusters.
    """
    (avg_cluster_dist,
     distance_to_closest,
     closest) = self._find_closest_cluster(cluster.center)
    if closest and distance_to_closest < merge_threshold:
      closest.merge(cluster)
    else:
      self._add_cluster(cluster)


  def _add_cluster(self, cluster):
    """
    Add cluster to the existing clusters.
    
    :param cluster: (Cluster) cluster to add.
    :raise: (ValueError) raise error if the cluster ID is already used. 
    """
    if cluster.id in self.clusters:
      raise ValueError('Cluster ID %s already exists' % cluster.id)
    self.clusters[cluster.id] = cluster


  def _find_closest_cluster(self, point):
    """
    Find the closest cluster to a point.
    
    :param point: (Point) The point of interest.
    :return average_cluster_distance: (float) average distance between point 
      and the centroids of all other clusters.
    :return distance_to_closest: (float) distance between closest cluster 
      center and point.
    :return closest: (Cluster) closest cluster to point.
    """
    distance_cluster_pairs = []
    for cluster in self.clusters.values():
      d = self.distance_func(cluster.center.value, point.value)
      distance_cluster_pairs.append((d, cluster))
    if len(distance_cluster_pairs) > 0:
      cluster_distances = [d[0] for d in distance_cluster_pairs]
      min_dist_idx = np.argmin(cluster_distances)

      # Get the closest cluster and some other useful distance metrics.
      average_cluster_distances = np.mean(cluster_distances)
      distance_to_closest = distance_cluster_pairs[min_dist_idx][0]
      closest = distance_cluster_pairs[min_dist_idx][1]
      return average_cluster_distances, distance_to_closest, closest
    else:
      return None, None, None


  @staticmethod
  def _noisy_sequence(anomaly_score, noisy_anomaly_score=0.3):
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
  def _stable_sequence(anomaly_score, stable_anomaly_score=0.2):
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



class PerfectClustering(OnlineClusteringV2):
  def __init__(self, distance_func):
    """
    This clustering implementation will return perfect cluster inferences.
    
    :param distance_func: (function) distance metric. The function signature 
      is "distance_func(p1, p2)" where p1 and p2 are instances of Point.
    :param merge_threshold: (float) cutoff distance to add or merge new 
      clusters.      
    """
    super(PerfectClustering, self).__init__(distance_func)


  def _find_closest_cluster(self, point):
    """
    Find the closest cluster to a point.
    
    :param point: (Point) The point of interest.
    :return distance_to_closest: (float) distance between closest cluster 
      center and point.
    :return closest: (Cluster) closest cluster to point.
    """
    for cluster in self.clusters.values():
      label_frequencies = cluster.label_distribution()
      most_frequent_cluster_label = label_frequencies[0]['label']
      if most_frequent_cluster_label == point.label:
        return 1.0, 0.0, cluster
    return None, None, None

# class OnlineClusteringV1(ClusteringInterface):
#   def cluster(self, sdr,  label=None):
#     """ See parent class docstring. """
# 
#     # If the sequence is stable (i.e. the anomaly score is low enough) then 
#     # find the closest cluster to the point. 
#     winning_cluster = None
#     if self.stable_sequence(anomaly_score):
#       point = Point(sdr, label)
#       if len(self.clusters) == 0:
#         winning_cluster = self.create_cluster(point)
#       else:
#         closest_cluster, distance_to_closest = self.find_closest_cluster(
#           point)
#         # If the distance to the closest cluster is low enough, then add the 
#         # point to the closest cluster. Otherwise, create a new cluster. 
#         if distance_to_closest < self.average_cluster_distance():
#           winning_cluster = closest_cluster
#         else:
#           winning_cluster = self.create_cluster(point)
#       winning_cluster.add(point)
# 
#     return winning_cluster
