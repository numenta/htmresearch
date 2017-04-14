# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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



def reshaped_sequence_distance(flattened_sequence_embeddings_1,
                               flattened_sequence_embeddings_2,
                               shape,
                               assume_sequence_alignment):
  sequence_embeddings_1 = flattened_sequence_embeddings_1.reshape(shape)
  sequence_embeddings_2 = flattened_sequence_embeddings_2.reshape(shape)
  return sequence_distance(sequence_embeddings_1, sequence_embeddings_2,
                           assume_sequence_alignment)



def sequence_distance(sequence_embeddings_1, sequence_embeddings_2,
                      assume_sequence_alignment):
  if assume_sequence_alignment:
    dists = aligned_distances(sequence_embeddings_1, sequence_embeddings_2)
  else:
    dists = min_distances(sequence_embeddings_1, sequence_embeddings_2)
  return np.mean(dists)



def min_distances(sequence_embeddings_1, sequence_embeddings_2):
  min_dists = []
  for e1 in sequence_embeddings_1:
    dists = []
    for e2 in sequence_embeddings_2:
      d = np.linalg.norm(e2 - e1)
      dists.append(d)
    min_dists.append(np.min(dists))
  return min_dists



def aligned_distances(sequence_embeddings_1, sequence_embeddings_2):
  if len(sequence_embeddings_1) != len(sequence_embeddings_2):
    raise ValueError('The two sequences need to have the same number of '
                     'embeddings. len(sequence_embeddings_1)=%s, '
                     'len(sequence_embeddings_2)=%s'
                     % (len(sequence_embeddings_1), len(sequence_embeddings_2)))

  aligned_dists = []
  for i in range(len(sequence_embeddings_1)):
    d = np.linalg.norm(sequence_embeddings_1[i] - sequence_embeddings_2[i])
    aligned_dists.append(d)
  return aligned_dists



def distance_matrix(sp_sequence_embeddings,
                    tm_sequence_embeddings, distance, sp_w=1.0, tm_w=1.0):
  if len(sp_sequence_embeddings) != len(tm_sequence_embeddings):
    raise ValueError('The number of SP sequence embeddings (%s) is '
                     'different from the number of TM sequence embeddings (%s)'
                     % (len(sp_sequence_embeddings),
                        len(tm_sequence_embeddings)))
  nb_sequences = len(sp_sequence_embeddings)
  col_mat = np.zeros((nb_sequences, nb_sequences), dtype=np.float64)
  cell_mat = np.zeros((nb_sequences, nb_sequences), dtype=np.float64)
  combined_mat = np.zeros((nb_sequences, nb_sequences), dtype=np.float64)

  for i in range(nb_sequences):
    for j in range(i, nb_sequences):
      col_dist = distance(sp_sequence_embeddings[i], sp_sequence_embeddings[j])
      cell_dist = distance(tm_sequence_embeddings[i], tm_sequence_embeddings[j])
      col_mat[i, j] = col_dist
      cell_mat[i, j] = cell_dist
      combined_mat[i, j] = (tm_w * col_dist + sp_w * cell_dist) / (
        tm_w + sp_w)

      col_mat[j, i] = col_mat[i, j]
      cell_mat[j, i] = cell_mat[i, j]
      combined_mat[j, i] = combined_mat[i, j]
  return col_mat, cell_mat, combined_mat



def euclidian_distance(x1, x2):
  return np.linalg.norm(np.array(x1) - np.array(x2))



def percent_overlap_distance(x1, x2):
  return 1 - percent_overlap(x1, x2)



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



def cluster_distance_matrix(sdr_clusters, distance_func):
  """
  Compute distance matrix between clusters of SDRs
  :param sdr_clusters: list of sdr clusters. Each cluster is a list of SDRs.
  :return: distance matrix
  """

  cluster_dist = cluster_distance_factory(distance_func)

  num_clusters = len(sdr_clusters)
  distance_mat = np.zeros((num_clusters, num_clusters), dtype=np.float64)

  for i in range(num_clusters):
    for j in range(i, num_clusters):
      distance_mat[i, j] = cluster_dist(sdr_clusters[i],
                                        sdr_clusters[j])
      distance_mat[j, i] = distance_mat[i, j]

  return distance_mat
