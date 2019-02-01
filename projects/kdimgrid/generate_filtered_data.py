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


def create_params(m, k, orthogonal, normalizeScales=True):
    B = np.zeros((m,k,k))
    A = np.zeros((m,2,k))

    S = 1 + np.random.normal(size=m, scale=0.2)

    if normalizeScales:
        S /= np.mean(S)

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


def processCubeQuery(query):
    phr, m, k, forceOrthogonal, normalizeScales = query

    upperBound = 4.0
    timeout = 60.0 * 10.0 # 10 minutes
    resultResolution = 0.01

    numDiscardedTooBig = 0
    numDiscardedTimeout = 0

    while True:
        expDict = create_params(m, k, forceOrthogonal, normalizeScales)

        # Rearrange to make the algorithms faster.
        sortOrder = np.argsort(expDict["S"])[::-1]
        expDict["S"] = expDict["S"][sortOrder]
        expDict["A"] = expDict["A"][sortOrder,:,:]

        try:
            binSidelength = computeBinSidelength(expDict["A"], phr,
                                                 resultResolution,
                                                 upperBound, timeout)

            if binSidelength == -1.0 or binSidelength >= 1.0:
                numDiscardedTooBig += 1
                continue

            expDict["bin_sidelength"] = binSidelength

            return (expDict, numDiscardedTooBig, numDiscardedTimeout)

        except RuntimeError as e:
            if e.message == "timeout":
                print "Timed out on query {}".format(expDict["A"])

                numDiscardedTimeout += 1
                continue
            else:
                raise



class Scheduler(object):
    def __init__(self, folderpath, numTrials, ms, ks, phaseResolutions,
                 allowOblique, normalizeScales):
        self.folderpath = folderpath
        self.numTrials = numTrials
        self.ms = ms
        self.ks = ks
        self.phaseResolutions = phaseResolutions

        self.failureCounter = 0
        self.successCounter = 0

        self.pool = multiprocessing.Pool()
        self.finishedEvent = threading.Event()

        forceOrthogonal = not allowOblique
        self.param_combinations = [(phr, m, k, forceOrthogonal, normalizeScales)
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
        self.pool.map_async(processCubeQuery, self.param_combinations,
                            callback=self.onWorkItemFinished)


    def onWorkItemFinished(self, results):
        discardedTooBig = np.full((len(self.phaseResolutions),
                                   len(self.ms),
                                   len(self.ks)),
                                  0, dtype="int")
        discardedTimeout = np.full((len(self.phaseResolutions),
                                    len(self.ms),
                                    len(self.ks)),
                                   0, dtype="int")
        binSidelengths = np.full((len(self.phaseResolutions),
                                  len(self.ms),
                                  len(self.ks)),
                                 np.nan, dtype="float")

        everyA = {}
        everyS = {}

        for params, result in zip(self.param_combinations, results):
            phr, m, k, _, _ = params
            expDict, numDiscardedTooBig, numDiscardedTimeout = result

            everyA[(phr, m, k)] = expDict["A"]
            everyS[(phr, m, k)] = expDict["S"]

            idx = (self.phaseResolutions.index(phr), self.ms.index(m),
                   self.ks.index(k))
            binSidelengths[idx] = expDict["bin_sidelength"]
            discardedTooBig[idx] += numDiscardedTooBig
            discardedTimeout[idx] += numDiscardedTimeout

        resultDict = {
            "phase_resolutions": self.phaseResolutions,
            "ms": self.ms,
            "ks": self.ks,
            "discarded_too_big": discardedTooBig,
            "discarded_timeout": discardedTimeout,
            "bin_sidelengths": binSidelengths,
            "every_A": everyA,
            "every_S": everyS,
        }

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folderName", type=str)
    parser.add_argument("--numTrials", type=int, default=1)
    parser.add_argument("--m", type=int, required=True, nargs="+")
    parser.add_argument("--k", type=int, required=True, nargs="+")
    parser.add_argument("--phaseResolution", type=float, default=[0.2], nargs="+")
    parser.add_argument("--allowOblique", action="store_true")
    parser.add_argument("--normalizeScales", action="store_true")

    args = parser.parse_args()

    cwd = os.path.dirname(os.path.realpath(__file__))
    folderpath = os.path.join(cwd, args.folderName)

    Scheduler(folderpath, args.numTrials, args.m, args.k, args.phaseResolution,
              args.allowOblique, args.normalizeScales).join()
