# ----------------------------------------------------------------------
#  Copyright (C) 2016, Numenta Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc. No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.
# ----------------------------------------------------------------------
import numpy as np
from clustering_interface import ClusteringInterface



class PerfectClustering(ClusteringInterface):
  def find_closest_cluster(self, point):
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
        return 0.0, cluster
    return None, None



class OnlineClusteringV2(ClusteringInterface):
  def find_closest_cluster(self, point):
    """
    Find the closest cluster to a point.
    
    :param point: (Point) The point of interest.
    :return distance_to_closest: (float) distance between closest cluster 
      center and point.
    :return closest: (Cluster) closest cluster to point.
    """
    distance_cluster_pairs = []
    for cluster in self.clusters.values():
      d = self.distance_func(cluster.center.value, point.value)
      distance_cluster_pairs.append((d, cluster))
    if len(distance_cluster_pairs) > 0:
      min_dist_idx = np.argmin([d[0] for d in distance_cluster_pairs])
      return distance_cluster_pairs[min_dist_idx]
    else:
      return None, None

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
