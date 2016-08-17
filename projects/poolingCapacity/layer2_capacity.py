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


def calculateNumCellsVsK(kVal, nVal, wVal, mVal):
  # number of columns
  n = Symbol("n", positive=True)
  # number of cells
  m = Symbol("m", positive=True)
  # number of connections per pattern
  w = Symbol("w", positive=True)
  # number of (feature, location) pairs
  k = Symbol("k", positive=True)

  numCellsInUnion = n * m * (1 - pow(1 - w / (n * m), k))

  numCellsInUnionVal = numCellsInUnion.subs(n, nVal).subs(w, wVal).subs(
    k, kVal).subs(m, mVal).evalf()

  return numCellsInUnionVal



def calculateSDRFalseMatchError(kVal,
                                thetaVal=20,
                                nVal=2048,
                                wVal=40,
                                mVal=10,
                                cVal=5):
  numCellsInUnionVal = calculateNumCellsVsK(kVal, nVal, cVal, mVal)

  pMatchBit = float(numCellsInUnionVal)/ (nVal * mVal)

  pFalseMatch = 1 - binom.cdf(thetaVal, wVal, pMatchBit)
  return pFalseMatch



def calculateObjectFalseMatchError(kVal, thetaVal=20, nVal=2048, wVal=40, mVal=10):
  pFalseMatchSDR = calculateSDRFalseMatchError(kVal, thetaVal, nVal, wVal, mVal)
  pFalseMatchObj = 1 - pow(1-pFalseMatchSDR, kVal)
  return pFalseMatchObj



def generateL4SDR(n=2048, m=10, w=40):
  colOrder = np.random.permutation(np.arange(n))
  activeCols = colOrder[:w]
  activeCells = np.random.randint(low=0, high=m, size=(w, ))

  activeBits = activeCols * m + activeCells
  return activeBits



def generateUnionSDR(k, n=2048, m=10, w=40, c=None):
  if c is None:
    c = w
  activeCells = set()
  connectedCells = set()
  for i in range(k):
    activeBits = generateL4SDR(n, m, w)
    activeBits = np.random.permutation(activeBits)
    activeCells = activeCells.union(activeBits)
    connectedCells = connectedCells.union(activeBits[:c])

  return connectedCells, activeCells



def generateMultipleUnionSDRs(numUnions, k, n=2048, m=10, w=40, c=None):
  if c is None:
    c = w
  activeCells = np.zeros((n * m, ))
  connectedCells = []
  for j in range(numUnions):
    connectedCells.append(np.zeros((n * m, )))

  for i in range(k):
    activeBits = generateL4SDR(n, m, w)
    activeCells[activeBits] = 1

    for j in range(numUnions):
      activeBits = np.random.permutation(activeBits)
      connectedCells[j][activeBits[:c]] = 1

  return connectedCells, activeCells



def simulateFalseMatchError(threshList, k, n=2048, w=40, m=10, c=10):
  connectedCells, activeCells = generateUnionSDR(k, n, m, w, c)

  numRpts = 10000
  numSDRMatch = []
  for rpt in range(numRpts):
    sdr = set(generateL4SDR(n, m, w))
    numSDRMatch.append(len(connectedCells.intersection(sdr)))

  numSDRMatch = np.array(numSDRMatch)
  falseMatchError = []
  for thresh in threshList:
    falseMatchError.append(
      np.sum(np.greater(numSDRMatch, thresh)).astype('float32') / numRpts)
  return falseMatchError



def simulateNumCellsVsK(kVal, nVal, wVal, mVal):
  connectedCells, activeCells = generateUnionSDR(kVal, nVal, mVal, wVal)
  numCellsInUnion = len(connectedCells)

  return numCellsInUnion



def simulateNumL2CellsFalseMatch(b1, n=2048, w=40, m=10, c=10, k=100):
  """
  Given an SDR that has theta bits overlap with one L2 cell,
  what is the chance that the SDR also has >theta bits overlap with
  a second L2 cell for this object?
  :param n: column # for L4
  :param w: active cell # for L4
  :param m: L4 cell # per column
  :param c: connectivity per L4 column
  :param k: (feature, location) # per object
  :return:
  """
  numRpts = 10000
  overlapList = np.zeros((numRpts,))

  for i in range(numRpts):
    (connectedCells, activeCells) = generateMultipleUnionSDRs(
      2, k, n, m, w, c)

    connectedCells1 = np.where(connectedCells[0])[0]
    nonConnectedCells1 = np.where(connectedCells[0]==0)[0]

    selectConnectedCell1 = connectedCells1[
      np.random.randint(0, len(connectedCells1), (theta, ))]
    selectNonConnectedCell1 = nonConnectedCells1[
      np.random.randint(0, len(nonConnectedCells1), (w-theta, ))]

    overlap = 0
    overlap += np.sum(connectedCells[1][selectConnectedCell1])
    overlap += np.sum(connectedCells[1][selectNonConnectedCell1])
    overlapList[i] = overlap

  return overlapList



def plotFalseMatchError(cValList, thetaValList):
  kValList = np.arange(0, 500, 10)

  fig, ax = plt.subplots(2, 1)
  colorList = ['r', 'm', 'g', 'b', 'c']

  legendList = []
  for i in range(len(cValList)):
    cVal = cValList[i]
    thetaVal = thetaValList[i]

    legendList.append('theta={}, c={}'.format(thetaVal, cVal))
    FalseMatchRateSDR = []
    numConnectedCells = []
    for kVal in kValList:
      FalseMatchRateSDR.append(calculateSDRFalseMatchError(
        kVal, thetaVal, nVal, wVal, mVal, cVal))

      numCellsInUnionVal = calculateNumCellsVsK(kVal, nVal, cVal, mVal)

      numConnectedCells.append(numCellsInUnionVal)
    ax[0].semilogy(kValList, FalseMatchRateSDR, colorList[i])
    ax[1].plot(kValList, numConnectedCells, colorList[i])


  ax[0].set_xlabel('# (feature, location)')
  ax[0].set_ylabel('SDR false match error')
  ax[0].set_ylim([pow(10, -13), 1])
  ax[0].legend(legendList, loc=4)

  ax[1].set_xlabel('# (feature, location)')
  ax[1].set_ylabel('# connections')



if __name__ == "__main__":

  nVal = 2048
  mVal = 10
  wVal = 40
  cVal = 10

  overlapList = simulateNumL2CellsFalseMatch(b1, nVal, wVal, mVal, cVal, k=100)

  fig, ax = plt.subplots(1, 1)
  legendList = []
  for cVal in [10, 20, 30, 40]:
    # theoretical values
    numCellsVsK = []
    kValList = np.arange(1, 500, 10)
    for kVal in kValList:
      numCellsInUnionVal = calculateNumCellsVsK(kVal, nVal, cVal, mVal)
      numCellsVsK.append(numCellsInUnionVal)
    legendList.append("c={}".format(cVal))
    ax.plot(kValList, numCellsVsK)

  for cVal in [10, 20, 30, 40]:
    # simulation values
    numCellsVsKsim = []
    kValListSparse = np.arange(1, 500, 100)
    for kVal in kValListSparse:
      numCellsInUnionValSim = simulateNumCellsVsK(kVal, nVal, cVal, mVal)
      numCellsVsKsim.append(numCellsInUnionValSim)
    ax.plot(kValListSparse, numCellsVsKsim, 'ko')

  ax.set_xlabel("# (feature, object) pair")
  ax.set_ylabel("# L4 inputs per L2 cell")
  ax.legend(legendList, loc=2)
  plt.savefig('UnionSizeVsK.pdf')

  plotFalseMatchError(cValList=[40, 40, 40, 40], thetaValList=[5, 10, 20, 30])
  plt.savefig('FalseMatchErrVsK_FixedCVaryingTheta.pdf')

  plotFalseMatchError(cValList=[10, 20, 30, 40], thetaValList=[10, 10, 10, 10])
  plt.savefig('FalseMatchErrVsK_FixedThetaVaryingC.pdf')

  plotFalseMatchError(cValList=[5, 10, 20, 30, 40], thetaValList=[3, 6, 12, 18, 24])
  plt.savefig('FalseMatchErrVsK_VaryingThetaandC.pdf')