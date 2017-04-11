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
from sklearn.manifold import TSNE, MDS



def project_clusters_2D(distance_mat, method='mds'):
  """
  Project SDRs onto a 2D space using manifold learning algorithms
  :param distance_mat: A square matrix with pairwise distances
  :param method: Select method from 'mds' and 'tSNE'
  :return: an array with dimension (numSDRs, 2). It contains the 2D projections
     of each SDR
  """
  seed = np.random.RandomState(seed=3)

  if method == 'mds':
    mds = MDS(n_components=2, max_iter=3000, eps=1e-9,
              random_state=seed,
              dissimilarity="precomputed", n_jobs=1)

    pos = mds.fit(distance_mat).embedding_

    nmds = MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
               dissimilarity="precomputed", random_state=seed,
               n_jobs=1, n_init=1)

    pos = nmds.fit_transform(distance_mat, init=pos)
  elif method == 'tSNE':
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    pos = tsne.fit_transform(distance_mat)
  else:
    raise NotImplementedError

  return pos



def project_matrix(mat):
  tsne = TSNE(n_iter=1000, metric='precomputed', init='random')
  return tsne.fit_transform(mat)



def project_vectors(vectors, distance):
  tsne = TSNE(metric=distance, n_iter=500, init='pca')
  return tsne.fit_transform(vectors)
