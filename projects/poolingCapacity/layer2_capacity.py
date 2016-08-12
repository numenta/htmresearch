# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

from sympy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import binom
mpl.rcParams['pdf.fonttype'] = 42

plt.ion()
plt.close('all')

"""
Analyze L4-L2 pooling capacity
"""


def calculateNumColsandCellsVsK(kVal, nVal, wVal, mVal):
  n = Symbol("n", positive=True)
  m = Symbol("m", positive=True)
  w = Symbol("w", positive=True)
  k = Symbol("k", positive=True)

  numCellsInUnion = n * m * (1 - pow(1 - w / (n * m), k))
  numColsInUnion = n * (1 - pow(1 - w / n, k))
  numCellsPerColumn = numCellsInUnion / numColsInUnion

  numColsInUnionVal = numColsInUnion.subs(n, nVal).subs(w, wVal).subs(
    k, kVal).evalf()

  numCellsInUnionVal = numCellsInUnion.subs(n, nVal).subs(w, wVal).subs(
    k, kVal).subs(m, mVal).evalf()

  numCellsPerColumnVal = numCellsPerColumn.subs(n, nVal).subs(w, wVal).subs(
    k, kVal).subs(m, mVal).evalf()

  return numColsInUnionVal, numCellsInUnionVal, numCellsPerColumnVal



def calculateSDRFalseMatchError(kVal, thetaVal=20, nVal=2048, wVal=40, mVal=32):
  (numColsInUnionVal,
   numCellsInUnionVal,
   numCellsPerColumnVal) = calculateNumColsandCellsVsK(kVal, nVal, wVal, mVal)

  pMatchBit = float((numColsInUnionVal / nVal) * (numCellsPerColumnVal / mVal))

  pFalseMatch = 1 - binom.cdf(thetaVal, wVal, pMatchBit)
  return pFalseMatch



def calculateObjectFalseMatchError(kVal, thetaVal=20, nVal=2048, wVal=40, mVal=32):
  pFalseMatchSDR = calculateSDRFalseMatchError(kVal, thetaVal, nVal, wVal, mVal)
  pFalseMatchObj = 1 - pow(1-pFalseMatchSDR, kVal)
  return pFalseMatchObj



def generateL4SDR(n=2048, m=32, w=40):
  colOrder = np.random.permutation(np.arange(n))
  activeCols = colOrder[:w]
  activeCells = np.random.randint(low=0, high=m, size=(w, ))

  activeBits = activeCols * m + activeCells
  return set(activeBits), set(activeCols)



def generateUnionSDR(k, n=2048, m=32, w=40):
  unionSDR = set()
  unionSDRcols = set()
  for i in range(k):
    activeBits, activeCols = generateL4SDR(n, m, w)
    unionSDR = unionSDR.union(activeBits)
    unionSDRcols = unionSDRcols.union(activeCols)
  return unionSDR, unionSDRcols



def simulateFalseMatchError(threshList, k, n=2048, w=40, m=32):
  unionSDR, unionSDRcols = generateUnionSDR(k, n, m, w)

  numRpts = 10000
  numSDRMatch = []
  for rpt in range(numRpts):
    sdr, sdrCols = generateL4SDR(n, m, w)
    numSDRMatch.append(len(unionSDR.intersection(sdr)))

  numSDRMatch = np.array(numSDRMatch)
  falseMatchError = []
  for thresh in threshList:
    falseMatchError.append(
      np.sum(np.greater(numSDRMatch, thresh)).astype('float32') / numRpts)
  return falseMatchError



def simulateNumColsandCellsVsK(kVal, nVal, wVal, mVal):
  activeBits, activeCols = generateUnionSDR(kVal, n=nVal, m=mVal, w=wVal)
  numColsInUnion = len(activeCols)
  numCellsInUnion = len(activeBits)
  numCellsPerColumn = float(numCellsInUnion) / numColsInUnion
  return numColsInUnion, numCellsInUnion, numCellsPerColumn



if __name__ == "__main__":

  nVal = 2048
  mVal = 32
  wVal = 40

  # theoretical values
  numCellsVsK, numColsVsK, numCellPerColumnVsK = [], [], []
  kValList = np.arange(1, 1000, 10)
  for kVal in kValList:
    (numColsInUnionVal,
     numCellsInUnionVal,
     numCellsPerColumnVal) = calculateNumColsandCellsVsK(
      kVal, nVal, wVal, mVal)

    numColsVsK.append(numColsInUnionVal)
    numCellsVsK.append(numCellsInUnionVal)
    numCellPerColumnVsK.append(numCellsPerColumnVal)

  # simulation values
  numCellsVsKsim, numColsVsKsim, numCellPerColumnVsKsim = [], [], []
  kValListSparse = np.arange(1, 1000, 100)
  for kVal in kValListSparse:
    (numColsInUnionValSim,
     numCellsInUnionValSim,
     numCellsPerColumnValSim) = simulateNumColsandCellsVsK(
      kVal, nVal, wVal, mVal)

    numColsVsKsim.append(numColsInUnionValSim)
    numCellsVsKsim.append(numCellsInUnionValSim)
    numCellPerColumnVsKsim.append(numCellsPerColumnValSim)

  fig, ax = plt.subplots(1, 2)
  ax[0].plot(kValList, numCellsVsK)
  ax[0].plot(kValList, numColsVsK, 'r')

  ax[0].plot(kValListSparse, numCellsVsKsim, 'bo')
  ax[0].plot(kValListSparse, numColsVsKsim, 'ro')

  ax[0].set_xlabel("# (feature, object) pair")
  ax[0].set_ylabel("# active bits in union")
  ax[0].legend(['Cell', 'Column'], loc=2)

  ax[1].plot(kValList, numCellPerColumnVsK)
  ax[1].plot(kValListSparse, numCellPerColumnVsKsim, 'bo')
  ax[1].set_xlabel("# (feature, object) pair")
  ax[1].set_ylabel("# cell per column")
  plt.savefig('UnionSizeVsK.pdf')


  kValList = np.arange(0, 1000, 10)
  fig, ax = plt.subplots(1, 2)
  colorList = ['r', 'm', 'g', 'b']
  i = 0
  for thetaVal in [10, 20, 30]:
    FalseMatchRateSDR = []
    FalseMatchRateObj = []
    for kVal in kValList:
      FalseMatchRateSDR.append(calculateSDRFalseMatchError(
        kVal, thetaVal, nVal, wVal, mVal))
      FalseMatchRateObj.append(calculateObjectFalseMatchError(
        kVal, thetaVal, nVal, wVal, mVal))

    ax[0].semilogy(kValList, FalseMatchRateSDR, colorList[i])
    ax[1].semilogy(kValList, FalseMatchRateObj, colorList[i])
    i += 1

  kValList = np.arange(0, 1000, 100)

  thetaList = [10, 20, 30]
  for kVal in kValList:
    falseMatchErr = simulateFalseMatchError(thetaList, kVal, nVal, wVal, mVal)
    for i in range(len(thetaList)):
      ax[0].semilogy(kVal, falseMatchErr[i], colorList[i]+'o')

  ax[0].set_xlabel('# (feature, location)')
  ax[1].set_xlabel('# (feature, location)')
  ax[0].set_ylabel('SDR false match error')
  ax[1].set_ylabel('Obj false match error')

  ax[0].set_ylim([pow(10, -13), 1])
  ax[1].set_ylim([pow(10, -13), 1])

  ax[0].legend(['theta=10', 'theta=20', 'theta=30'], loc=4)
  ax[1].legend(['theta=10', 'theta=20', 'theta=30'], loc=4)
  plt.savefig('FalseMatchErrVsK.pdf')