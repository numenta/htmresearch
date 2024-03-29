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
import os
import pickle
import re

import numpy as np

from htmresearch_core.experimental import computeGridUniquenessHypercube


def create_L(m, theta=np.pi/3.):
    L = np.zeros((m,2,2))
    for i in range(m):
      L[i] = np.array([
          [np.cos(0.), np.cos(theta)],
          [np.sin(0.), np.sin(theta)]
        ])
    return L


def extractNumber(filename):
    return int(re.match("in_([0-9]+).p", filename).groups()[0])


def measureSidelengths(folderPath):
    inPath = os.path.join(folderPath, "in")
    outPath = os.path.join(folderPath, "out")

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    files = sorted((filename
                    for filename in os.listdir(inPath)
                    if re.match("in_[0-9]+.p", filename)),
                   key=extractNumber)

    for filename in files:
        inFilePath = os.path.join(folderPath, "in", filename)

        iFile = int(re.match("in_([0-9]+).p", filename).groups()[0])

        print "Computing result", iFile

        outFilePath = os.path.join(folderPath, "out", "res_{}.p".format(iFile))

        with open(inFilePath, "r") as fin:
            result_dict = pickle.load(fin)

        ms = result_dict["ms"]
        ks = result_dict["ks"]
        phase_resolutions = result_dict["phase_resolutions"]
        S = result_dict["S"]
        A = result_dict["A"]
        bin_sidelengths = result_dict["bin_sidelength"]
        L = create_L(max(ms))

        param_combinations = [(phr, m, k)
                              for phr in phase_resolutions
                              for m in ms
                              for k in ks
                              if 2*m >= k]

        unique_sidelengths = np.full_like(bin_sidelengths, np.nan)

        for phr, m, k in param_combinations:
            A_ = A[:m, :, :k]
            sort_order = np.argsort(S[:m])[::-1]
            A_ = A_[sort_order, :, :]
            L_ = L[:m]

            bin_sidelength = bin_sidelengths[phase_resolutions.index(phr),
                                             ms.index(m), ks.index(k)]

            unique_sidelength, _ = computeGridUniquenessHypercube(
                A_, L_, phr, ignoredCenterDiameter=bin_sidelength/2)

            unique_sidelengths[phase_resolutions.index(phr),
                               ms.index(m), ks.index(k)] = unique_sidelength

        result_dict["width"] = unique_sidelengths

        with open(outFilePath, "w") as fout:
            print "Saving", outFilePath
            pickle.dump(result_dict, fout)


def measureNormalizedSidelengths(folderPath):
    inPath = os.path.join(folderPath, "in")
    outPath = os.path.join(folderPath, "out")

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    files = sorted((filename
                    for filename in os.listdir(inPath)
                    if re.match("in_[0-9]+.p", filename)),
                   key=extractNumber)

    for filename in files:
        inFilePath = os.path.join(folderPath, "in", filename)

        iFile = int(re.match("in_([0-9]+).p", filename).groups()[0])

        print "Computing result", iFile

        outFilePath = os.path.join(folderPath, "out", "res_{}.p".format(iFile))

        with open(inFilePath, "r") as fin:
            result_dict = pickle.load(fin)

        ms = result_dict["ms"]
        ks = result_dict["ks"]
        phase_resolutions = result_dict["phase_resolutions"]
        rectangles = result_dict["rectangles"]

        A = result_dict["A"]
        L = create_L(max(ms))

        unique_sidelengths = np.full((len(phase_resolutions),
                                      len(ms), len(ks)),
                                     np.nan, dtype="float")

        for (phr, m, k), rectangle in rectangles.iteritems():
            A_ = A[:m, :, :k].copy()
            for iDim in xrange(k):
                A_[:,:,iDim] *= rectangle[iDim]

            L_ = L[:m]

            # Normalizing puts the edge of the bin at approximately 0.5.
            unique_sidelength, _ = computeGridUniquenessHypercube(
                A_, L_, phr,
                ignoredCenterDiameter=0.51)

            unique_sidelengths[phase_resolutions.index(phr),
                               ms.index(m), ks.index(k)] = unique_sidelength

        result_dict["width"] = unique_sidelengths

        with open(outFilePath, "w") as fout:
            print "Saving", outFilePath
            pickle.dump(result_dict, fout)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("folderName", type=str)
    parser.add_argument("--normalizeBasis", action="store_true")

    args = parser.parse_args()

    cwd = os.path.dirname(os.path.realpath(__file__))
    folderPath = os.path.join(cwd, args.folderName)

    if args.normalizeBasis:
        measureNormalizedSidelengths(folderPath)
    else:
        measureSidelengths(folderPath)
