# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import argparse
import multiprocessing
import os
import pickle
import time

import numpy as np
from scipy.stats import ortho_group

from htmresearch_core.experimental import computeBinSidelength


def create_orthogonal_projection_base(k, s):
    assert k > 1
    B = np.eye(k)
    Q = ortho_group.rvs(k)

    return np.dot(Q,s*B)


def random_point_on_circle():
    r = np.random.sample()*2.*np.pi
    return np.array([np.cos(r), np.sin(r)])


def create_params(m,k):
    B = np.zeros((m,k,k))
    A = np.zeros((m,2,k))
    # S = np.sqrt(2)**np.arange(m)
    S = np.ones(m) + np.random.sample(m)

    for m_ in range(m):
        if k==1:
            B[m_,0,0] = S[m_]*np.random.choice([-1.,1.])
            A[m_] = np.dot(random_point_on_circle().reshape((2,1)), np.linalg.inv(B[m_]))
        else:
            B[m_] = create_orthogonal_projection_base(k, S[m_])
            A[m_] = np.linalg.inv(B[m_])[:2]

    return {
        "A": A,
        "S": S,
    }


class ResultsForSingleMatrix(object):
    def __init__(self, result_dict, mk_combinations, ms, ks, file_path, all_file_paths):
        self.result_dict = result_dict
        self.mk_combinations = mk_combinations
        self.ms = ms
        self.ks = ks
        self.file_path = file_path
        self.all_file_paths = all_file_paths

    def onFinished(self, results):
        """
        Write a file now that everything is finished.

        @param (result)
        List of bin sidelengths.
        """

        bin_sidelengths = np.full((len(self.ms), len(self.ks)),
                                  np.nan, dtype="float")

        for (m, k), result in zip(self.mk_combinations, results):
            bin_sidelengths[self.ms.index(m), self.ks.index(k)] = result

        self.result_dict["bin_sidelength"] = bin_sidelengths

        with open(self.file_path, "w") as fout:
            self.all_file_paths.remove(self.file_path)
            print "Saving", self.file_path, "({} remaining)".format(
                len(self.all_file_paths))
            pickle.dump(self.result_dict, fout)


def getQuery(A, S, m, k, phase_resolution):
    A_ = A[:m, :, :k]
    sort_order = np.argsort(S[:m])[::-1]
    A_ = A_[sort_order, :, :]

    return (A_, phase_resolution)


def processQuery(query):
    A, phase_resolution = query
    return computeBinSidelength(A, phase_resolution, 0.01)


def generateData(folderPath, numTrials, ms, ks,
                 phaseResolution=0.2):
    pool = multiprocessing.Pool()

    mk_combinations = [(m, k)
                       for m in ms
                       for k in ks
                       if 2*m >= k]

    async_rs = []

    all_file_paths = set()

    for i in xrange(numTrials):
        result_dict = create_params(max(ms), max(ks))
        result_dict["phase_resolution"] = phaseResolution
        result_dict["ms"] = ms
        result_dict["ks"] = ks
        A = result_dict["A"]
        S = result_dict["S"]

        filename = os.path.join(folderPath, "in_{}.p".format(i))

        all_file_paths.add(filename)

        resultSaver = ResultsForSingleMatrix(result_dict, mk_combinations,
                                             ms, ks, filename, all_file_paths)
        async_rs.append(
            pool.map_async(processQuery,
                           (getQuery(A, S, m, k, phaseResolution)
                            for m, k in mk_combinations),
                           callback=resultSaver.onFinished))

    pool.close()

    # Don't use pool.join() to wait for threads. multiprocessing doesn't
    # properly handle interrupts (ctrl+c) during pool.join().
    try:
        for r in async_rs:
            r.get(9999999)
        pool.join()
    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folderName", type=str)
    parser.add_argument("--numTrials", type=int, default=1)
    parser.add_argument("--m", type=int, required=True, nargs="+")
    parser.add_argument("--k", type=int, required=True, nargs="+")

    args = parser.parse_args()

    cwd = os.path.dirname(os.path.realpath(__file__))
    folderPath = os.path.join(cwd, "data", args.folderName, "in")

    os.makedirs(folderPath)

    generateData(folderPath, args.numTrials, args.m, args.k)
