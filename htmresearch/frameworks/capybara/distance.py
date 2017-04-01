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
