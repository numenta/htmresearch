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

from htmresearch_core.experimental import (computeBinSidelength,
                                           computeBinRectangle)


def create_bases(k, s):
    assert(k>1)
    B = np.zeros((k,k))
    B[:2,:2] = np.array([
        [1.,0.],
        [0.,1.]
    ])
    for col in range(2,k):
        B[ :, col] = np.random.randn(k)
        B[ :, col] = B[ :, col]/np.linalg.norm(B[ :, col])

    Q = ortho_group.rvs(k)
    return np.dot(Q,s*B)


def create_orthogonal_projection_base(k, s):
    assert k > 1
    B = np.eye(k)
    Q = ortho_group.rvs(k)

    return np.dot(Q,s*B)


def random_point_on_circle():
    r = np.random.sample()*2.*np.pi
    return np.array([np.cos(r), np.sin(r)])


def create_params(m, k, orthogonal):
    B = np.zeros((m,k,k))
    A = np.zeros((m,2,k))
    # S = np.sqrt(2)**np.arange(m)
    S = np.ones(m) + np.random.sample(m)

    for m_ in range(m):
        if k==1:
            B[m_,0,0] = S[m_]*np.random.choice([-1.,1.])
            A[m_] = np.dot(random_point_on_circle().reshape((2,1)), np.linalg.inv(B[m_]))
        else:
            if orthogonal:
                B[m_] = create_orthogonal_projection_base(k, S[m_])
            else:
                B[m_] = create_bases(k, S[m_])
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


def processCubeQuery(query):
    A, phase_resolution = query
    resultResolution = 0.01
    upperBound = 2048.0
    timeout = 60.0 * 10.0 # 10 minutes

    try:
        result = computeBinSidelength(A, phase_resolution, resultResolution,
                                      upperBound, timeout)
        if result == -1.0:
            print "Couldn't find bin smaller than {} for query {}".format(
                upperBound, A.tolist())
            return None

        return result
    except RuntimeError as e:
        if e.message == "timeout":
            print "Timed out on query {}".format(A.tolist())
            return None
        else:
            raise


def processRectangleQuery(query):
    A, phase_resolution = query
    resultResolution = 0.01
    upperBound = 2048.0
    timeout = 60.0 * 10.0 # 10 minutes

    try:
        result = computeBinRectangle(A, phase_resolution, resultResolution,
                                     upperBound, timeout)

        if len(result) == 0:
            print "Couldn't find bin smaller than {} for query {}".format(
                upperBound, A.tolist())
            return None

        return result
    except RuntimeError as e:
        if e.message == "timeout":
            print "Timed out on query {}".format(A.tolist())
            return None
        else:
            raise



class IterableWithLen(object):
    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.iterable



class Scheduler(object):
    def __init__(self, folderpath, numTrials, ms, ks, phaseResolutions,
                 measureRectangle, allowOblique):
        self.folderpath = folderpath
        self.numTrials = numTrials
        self.ms = ms
        self.ks = ks
        self.phaseResolutions = phaseResolutions
        self.measureRectangle = measureRectangle
        self.allowOblique = allowOblique

        self.failureCounter = 0
        self.successCounter = 0

        self.pool = multiprocessing.Pool()
        self.finishedEvent = threading.Event()

        self.param_combinations = [(phr, m, k)
                                   for phr in phaseResolutions
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
        forceOrthogonal = not self.allowOblique
        resultDict = create_params(max(self.ms), max(self.ks), forceOrthogonal)
        resultDict["phase_resolutions"] = self.phaseResolutions
        resultDict["ms"] = self.ms
        resultDict["ks"] = self.ks

        A = resultDict["A"]
        S = resultDict["S"]

        queries = (getQuery(A, S, m, k, phr)
                   for phr, m, k in self.param_combinations)
        # map_async will convert this to a list if it can't get the length.
        queries = IterableWithLen(queries, len(self.param_combinations))

        if self.measureRectangle:
            operation = processRectangleQuery
        else:
            operation = processCubeQuery

        context = ContextForSingleMatrix(self, resultDict)
        self.pool.map_async(operation, queries, callback=context.onFinished)


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
        if self.measureRectangle:
            rectangles = {}
            for (phr, m, k), result in zip(self.param_combinations, results):
                rectangles[(phr, m, k)] = result

            resultDict["rectangles"] = rectangles
        else:
            bin_sidelengths = np.full((len(self.phaseResolutions),
                                       len(self.ms),
                                       len(self.ks)),
                                      np.nan, dtype="float")
            for (phr, m, k), result in zip(self.param_combinations, results):
                bin_sidelengths[self.phaseResolutions.index(phr), self.ms.index(m),
                                self.ks.index(k)] = result
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
    parser.add_argument("--phaseResolution", type=float, default=[0.2], nargs="+")
    parser.add_argument("--measureRectangle", action="store_true")
    parser.add_argument("--allowOblique", action="store_true")

    args = parser.parse_args()

    cwd = os.path.dirname(os.path.realpath(__file__))
    folderpath = os.path.join(cwd, args.folderName)

    Scheduler(folderpath, args.numTrials, args.m, args.k, args.phaseResolution,
              args.measureRectangle, args.allowOblique).join()
