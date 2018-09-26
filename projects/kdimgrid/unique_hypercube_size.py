# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

from collections import defaultdict
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from htmresearch_core.experimental import computeGridUniquenessHypercube

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def doRandomModuleExperiment(ms, ks, latticeAngle=np.radians(90)):
  scales = [1.*(math.sqrt(2)**s) for s in xrange(max(ms))]
  readoutResolution = 0.2
  kmax = max(ks)

  domainToPlaneByModule = []
  latticeBasisByModule = []
  for s in xrange(max(ms)):
    b1 = np.random.multivariate_normal(mean=np.zeros(kmax), cov=np.eye(kmax))
    b1 /= np.linalg.norm(b1)

    # Choose a random vector orthogonal to b1
    while True:
      randomVector = np.random.multivariate_normal(mean=np.zeros(kmax), cov=np.eye(kmax))
      randomVector /= np.linalg.norm(randomVector)
      projectedToPlane = randomVector - np.dot(randomVector, b1) * b1

      length = np.linalg.norm(projectedToPlane)
      if length == 0:
        continue

      b2 = projectedToPlane / length
      break

    # Choose a set of other random basis vectors
    bases = np.zeros((kmax, kmax), dtype="float")
    bases[:,0] = b1
    bases[:,1] = b2
    for iDim in xrange(2, kmax):
        b = np.random.multivariate_normal(mean=np.zeros(kmax), cov=np.eye(kmax))
        b /= np.linalg.norm(b)
        bases[:,iDim] = b

    scale = math.sqrt(2)**s
    domainToPlaneByModule.append(np.linalg.inv(scale*bases)[:2])
    latticeBasisByModule.append([[math.cos(0.), math.cos(latticeAngle)],
                                 [math.sin(0.), math.sin(latticeAngle)]])

  domainToPlaneByModule = np.array(domainToPlaneByModule, dtype="float")
  latticeBasisByModule = np.array(latticeBasisByModule, dtype="float")

  results = {}

  for m in ms:
    for k in ks:
      domainToPlaneByModule_ = domainToPlaneByModule[:m,:,:k]
      latticeBasisByModule_ = latticeBasisByModule[:m]
      print "domainToPlaneByModule", domainToPlaneByModule_
      result = computeGridUniquenessHypercube(domainToPlaneByModule_,
                                              latticeBasisByModule_,
                                              readoutResolution, 0.5)
      results[(m, k)] = result[0]

  return domainToPlaneByModule, results


def experiment1():
  ms = range(1, 8)
  ks = range(1, 7)
  numTrials = 1

  allResultsByParams = defaultdict(list)
  for _ in xrange(numTrials):
    A, resultsByParams = doRandomModuleExperiment(ms, ks)
    for params, v in resultsByParams.iteritems():
      allResultsByParams[params].append(v)

  meanResultByParams = {}
  for params, listOfResults in allResultsByParams.iteritems():
    meanResultByParams[params] = (sum(listOfResults) / len(listOfResults))

  timestamp = time.strftime("%Y%m%d-%H%M%S")

  # Diameter plot
  plt.figure()
  for m in ms:
    x = []
    y = []
    for k in ks:
      x.append(k)
      y.append(meanResultByParams[(m,k)])
    plt.plot(x, y, marker='o')
  plt.yscale('log')
  plt.xticks(ks)
  plt.xlabel("Number of dimensions")
  plt.ylabel("Diameter of unique hypercube")
  plt.legend(["{} module{}".format(m, "" if m == 0 else "s")
              for m in ms])
  filename = "Diameter_%s.pdf" % timestamp
  filePath = os.path.join(CHART_DIR, filename)
  print "Saving", filePath
  plt.savefig(filePath)

  # Volume plot
  plt.figure()
  for m in ms:
    x = []
    y = []
    for k in ks:
      x.append(k)
      y.append(math.pow(meanResultByParams[(m,k)], k))
    plt.plot(x, y, marker='o')
  plt.yscale('log')
  plt.xticks(ks)
  plt.xlabel("Number of dimensions")
  plt.ylabel("Volume of unique hypercube")
  plt.legend(["{} module{}".format(m, "" if m == 0 else "s")
              for m in ms])
  filename = "Volume_%s.pdf" % timestamp
  filePath = os.path.join(CHART_DIR, filename)
  print "Saving", filePath
  plt.savefig(filePath)


if __name__ == "__main__":
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  experiment1()
