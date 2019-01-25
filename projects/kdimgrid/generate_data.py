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
import threading

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


def getQuery(A, S, m, k, phase_resolution):
    A_ = A[:m, :, :k]
    sort_order = np.argsort(S[:m])[::-1]
    A_ = A_[sort_order, :, :]

    return (A_, phase_resolution)


def processQuery(query):
    A, phase_resolution = query

    try:
        timeout = 60.0 * 10.0 # 10 minutes
        return computeBinSidelength(A, phase_resolution, 0.01, timeout=timeout)
    except RuntimeError as e:
        if e.message == "timeout":
            print "Timed out on query {}".format(A.tolist())
            return None
        else:
            raise


class Scheduler(object):
    def __init__(self, folderpath, numTrials, ms, ks, phaseResolution=0.2):
        self.folderpath = folderpath
        self.numTrials = numTrials
        self.ms = ms
        self.ks = ks
        self.phaseResolution = phaseResolution
        self.failureCounter = 0
        self.successCounter = 0

        self.pool = multiprocessing.Pool()
        self.finishedEvent = threading.Event()

        self.mk_combinations = [(m, k)
                                for m in ms
                                for k in ks
                                if 2*m >= k]

        for _ in xrange(numTrials):
            self.queueNewWorkItem()


    def join(self):
        try:
            # Interrupts (ctrl+c) have no effect without a timeout.
            self.finishedEvent.wait(9999999999)
            self.pool.close()
            self.pool.join()
        except KeyboardInterrupt:
            print "Caught KeyboardInterrupt, terminating workers"
            self.pool.terminate()
            self.pool.join()


    def queueNewWorkItem(self):
        resultDict = create_params(max(self.ms), max(self.ks))
        resultDict["phase_resolution"] = self.phaseResolution
        resultDict["ms"] = self.ms
        resultDict["ks"] = self.ks

        A = resultDict["A"]
        S = resultDict["S"]
        queries = (getQuery(A, S, m, k, self.phaseResolution)
                   for m, k in self.mk_combinations)

        context = ContextForSingleMatrix(self, resultDict)

        self.pool.map_async(processQuery, queries, callback=context.onFinished)


    def handleFailure(self, resultDict):
        failureFolder = os.path.join(self.folderpath, "failures")
        if self.failureCounter == 0:
            os.makedirs(failureFolder)

        filename = "failure_{}.p".format(self.failureCounter)
        self.failureCounter += 1

        filepath = os.path.join(failureFolder, filename)

        with open(filepath, "w") as fout:
            print "Saving", filepath, "({} remaining)".format(
                self.numTrials - self.successCounter)
            pickle.dump(resultDict, fout)


        self.queueNewWorkItem()


    def handleSuccess(self, resultDict, results):
        # Insert results into dict
        bin_sidelengths = np.full((len(self.ms), len(self.ks)),
                                  np.nan, dtype="float")
        for (m, k), result in zip(self.mk_combinations, results):
            bin_sidelengths[self.ms.index(m), self.ks.index(k)] = result
        resultDict["bin_sidelength"] = bin_sidelengths

        # Save the dict
        successFolder = os.path.join(self.folderpath, "in")
        if self.successCounter == 0:
            os.makedirs(successFolder)
        filepath = os.path.join(successFolder, "in_{}.p".format(
            self.successCounter))
        self.successCounter += 1
        with open(filepath, "w") as fout:
            print "Saving", filepath, "({} remaining)".format(
                self.numTrials - self.successCounter)
            pickle.dump(resultDict, fout)

        if self.successCounter == self.numTrials:
            self.finishedEvent.set()


class ContextForSingleMatrix(object):
    def __init__(self, scheduler, resultDict):
        self.scheduler = scheduler
        self.resultDict = resultDict

    def onFinished(self, results):
        if any(result is None
               for result in results):
            self.scheduler.handleFailure(self.resultDict)
        else:
            self.scheduler.handleSuccess(self.resultDict, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folderName", type=str)
    parser.add_argument("--numTrials", type=int, default=1)
    parser.add_argument("--m", type=int, required=True, nargs="+")
    parser.add_argument("--k", type=int, required=True, nargs="+")

    args = parser.parse_args()

    cwd = os.path.dirname(os.path.realpath(__file__))
    folderpath = os.path.join(cwd, "data", args.folderName)

    Scheduler(folderpath, args.numTrials, args.m, args.k).join()
