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


def percent_overlap(x1, x2):
  """
  Computes the percentage of overlap between SDR 1 and 2

  :param x1: (np.array) binary vector 1
  :param x2: (np.array) binary vector 2

  :return pct_overlap: (float) percentage overlap between SDR 1 and 2
  """
  if type(x1) is np.ndarray and type(x2) is np.ndarray:
    non_zero_1 = float(np.count_nonzero(x1))
    non_zero_2 = float(np.count_nonzero(x2))
    min_non_zero = min(non_zero_1, non_zero_2) 
    pct_overlap = 0
    if min_non_zero > 0:
      pct_overlap = float(np.dot(x1, x2)) / np.sqrt(non_zero_1 * non_zero_2)
  else:
    raise ValueError("x1 and x2 need to be binary numpy array but are: "
                     "%s" % type(x1))

  return pct_overlap

def euclidian_distance(x1, x2):
  return np.linalg.norm(np.array(x1) - np.array(x2))

def percent_overlap_distance(x1, x2):
  return 1 - percent_overlap(x1, x2)

def cluster_distance_factory(distance):
  def cluster_distance(c1, c2):
    """
    Symmetric distance between two clusters
  
    :param c1: (np.array) cluster 1
    :param c2: (np.array) cluster 2
    :return: distance between 2 clusters
    """
    cluster_dist = cluster_distance_directed_factory(distance)
    d12 = cluster_dist(c1, c2)
    d21 = cluster_dist(c2, c1)
    return np.mean([d12, d21])


  return cluster_distance



def cluster_distance_directed_factory(distance):
  def cluster_distance_directed(c1, c2):
    """
    Directed distance from cluster 1 to cluster 2
  
    :param c1: (np.array) cluster 1
    :param c2: (np.array) cluster 2
    :return: distance between 2 clusters
    """
    if len(c1) == 0 or len(c2) == 0:
      return 0
    else:
      return distance(np.sum(c1, axis=0) / float(len(c1)),
                      np.sum(c2, axis=0) / float(len(c2)))


  return cluster_distance_directed
