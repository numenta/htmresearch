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
from scipy.stats import entropy


def pairwise_entropy(Y):
    n = Y.shape[0]
    
    P = np.zeros((4, n,n))
    
    P[0] = np.mean(np.expand_dims(Y , axis=1) + np.expand_dims(Y , axis=0) == 0, axis=2)
    P[1] = np.mean(np.expand_dims(Y , axis=1) > np.expand_dims(Y , axis=0)     , axis=2)
    P[2] = np.mean(np.expand_dims(Y , axis=1) < np.expand_dims(Y , axis=0)     , axis=2)
    P[3] = np.mean(np.expand_dims(Y , axis=1) + np.expand_dims(Y , axis=0) == 2, axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        P_dot_logP = np.where(P==0, 0, P*np.log2(P))

    pairwise_H = - np.sum(P_dot_logP, axis=0)
    return pairwise_H



def mean_mutual_info(P):

    mu    = 0.0
    count = 0.0
    for (i, j), pij in np.ndenumerate(P):
        if i != j:
            count += 1
            pi = P[i,i]
            pj = P[j,j]
            q = [pij, pj - pij, pi - pij, 1 + pij - pi - pj]

            mu += entropy([pi, 1-pi], base=2) + entropy([pj, 1-pj], base=2) - entropy(q, base=2)

    return mu/float(count)


def mean_mutual_info_from_data(Y):
    (n, d) = Y.shape
    A = np.expand_dims(Y, axis=1) * np.expand_dims(Y, axis=0)
    Q = np.mean(A, axis=2)
    Q[np.where(Q == 0.)] = 0.000001

    mean_info = mean_mutual_info(Q)

    return mean_info


def mean_mutual_info_from_model(pooler):
    Q = pooler.avg_activity_pairs
    mean_info = mean_mutual_info(Q)

    return mean_info


