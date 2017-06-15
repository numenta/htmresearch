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
  """ Generate single  L4 SDR, return active bits"""
  colOrder = np.random.permutation(np.arange(n))
  activeCols = colOrder[:w]
  activeCells = np.random.randint(low=0, high=m, size=(w, ))

  activeBits = activeCols * m + activeCells
  return activeBits



def generateUnionSDR(k, n=2048, m=10, w=40, c=None):
  """ Generate a set of L4 cells that are connected to a L2 neuron """
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



def generateMultipleUnionSDRs(numL2Cell, k, n=2048, m=10, w=40, c=None):
  """ Generate numL2Cell set of L4 cells that are connected to numL2Cell
  L2 cells
  """
  if c is None:
    c = w
  activeCells = np.zeros((n * m, ))
  connectedCells = []
  for j in range(numL2Cell):
    connectedCells.append(np.zeros((n * m, )))

  for i in range(k):
    activeBits = generateL4SDR(n, m, w)
    activeCells[activeBits] = 1

    for j in range(numL2Cell):
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



def simulateL2CellPairsConditionalFalseMatch(b1, n=2048, w=40, m=10, c=10, k=100):
  """
  Given an SDR that has b1 bits overlap with one L2 cell,
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
      np.random.randint(0, len(connectedCells1), (b1, ))]
    selectNonConnectedCell1 = nonConnectedCells1[
      np.random.randint(0, len(nonConnectedCells1), (w-b1, ))]

    overlap = 0
    overlap += np.sum(connectedCells[1][selectConnectedCell1])
    overlap += np.sum(connectedCells[1][selectNonConnectedCell1])
    overlapList[i] = overlap
  return overlapList



def simulateL2CellPairFalseMatch(theta, n=2048, w=40, m=10, c=10, k=100):
  numRpts = 100000
  numL2cell = 2

  numMatchPair = 0
  numMatch = np.zeros((numL2cell, ))
  for _ in range(numRpts):
    (connectedCells, activeCells) = generateMultipleUnionSDRs(
      numL2cell, k, n, m, w, c)
    l4SDR = generateL4SDR(n, m, w)

    match = 1
    for j in range(numL2cell):
      if np.sum(connectedCells[j][l4SDR]) <= theta:
        match = 0
      else:
        numMatch[j] += 1

    numMatchPair += match


def simulateL4L2Pooling(theta=4, n=150, w=10, m=16, c=5, k=100):
  numRpts = 1000
  numL2cell = 40

  calculateNumCellsVsK(k, n, c, m)
  numMatch = np.zeros((numRpts, numL2cell))
  for i in range(numRpts):
    print i
    (connectedCells, activeCells) = generateMultipleUnionSDRs(
      numL2cell, k, n, m, w, c)
    l4SDR = generateL4SDR(n, m, w)

    for j in range(numL2cell):
      if np.sum(connectedCells[j][l4SDR]) > theta:
        numMatch[i, j] += 1

  plt.figure()
  falseCells = np.sum(numMatch, 1)
  binwidth = 1
  plt.hist(falseCells, bins=np.arange(min(falseCells)-.5, max(falseCells) + .5, binwidth))
  plt.xlabel('falsely activated output cells #')
  plt.ylabel('Frequency ')
  plt.savefig('L4L2PoolingSimulation.pdf')


def computeL2CellPairsFalseMatchChance(thetaVal, nVal, mVal, wVal, kVal, cVal):
  n = Symbol("n", positive=True)
  m = Symbol("m", positive=True)
  w = Symbol("w", positive=True)
  k = Symbol("k", positive=True)
  c = Symbol("c", positive=True)
  b1 = Symbol("b1", positive=True)
  b2 = Symbol("b2", positive=True)

  numOverlap = n * m * (1 - pow(1 - (c * c) / (w * n * m), k))
  numCellsInUnion = n * m * (1 - pow(1 - c / (n * m), k))
  numTotal = binomial(n*m - numCellsInUnion, w-b1) * binomial(numCellsInUnion, b1)

  numCellsInUnionVal = int(numCellsInUnion.subs(k, kVal).subs(c, cVal).\
    subs(n, nVal).subs(m, mVal).evalf())

  numOverlapVal = int(
    numOverlap.subs(k, kVal).subs(c, cVal).subs(n, nVal).subs(m, mVal).subs(
      w, wVal).evalf())

  pFalseMatchPair = 0

  pFalseMatchb1ValDict = {}
  pFalseMatchb1b2Dict  = {}
  pFalseMatchb2Givenb1Dict = {}
  for b1Val in range(thetaVal+1, wVal+1):
    p = numCellsInUnion / (n * m)
    pFalseMatchb1 = binomial(w, b1) * pow(p, b1) * pow(1 - p, w - b1)
    pFalseMatchb1Val = pFalseMatchb1.subs(b1, b1Val).subs(n, nVal).\
      subs(m, mVal).subs(k, kVal).subs(w, wVal).subs(c, cVal).evalf()

    numTotalVal = numTotal.subs(n, nVal).subs(m, mVal). \
      subs(k, kVal).subs(w, wVal).subs(c, cVal).subs(b1, b1Val).evalf()

    pFalseMatchb1ValDict[b1Val] = pFalseMatchb1Val

    for b2Val in range(thetaVal+1, wVal+1):

      minI = max(max(b1Val - (numCellsInUnionVal - numOverlapVal), 0),
                 max(b2Val - (numCellsInUnionVal - numOverlapVal), 0),
                 max(b1Val + b2Val - wVal, 0))

      maxI = min(b1Val, b2Val)

      if minI > maxI:
        continue
      numMatchPair = 0
      for i in range(minI, maxI+1):
        numMatchPair += (binomial(numOverlapVal, i) *
                         binomial(numCellsInUnionVal - numOverlapVal, b2Val - i) *
                         binomial(numCellsInUnionVal - numOverlapVal, b1Val - i) *
                         binomial(nVal*mVal-2*numCellsInUnionVal + numOverlapVal,
                                  wVal-b1Val-b2Val+i))

      pFalseMatchb2Givenb1 = (numMatchPair / numTotalVal)

      pFalseMatchb2Givenb1Dict[(b1Val, b2Val)] = pFalseMatchb2Givenb1
      pFalseMatchb1b2Dict[(b1Val, b2Val)] = pFalseMatchb2Givenb1 * pFalseMatchb1Val
      pFalseMatchPair += pFalseMatchb2Givenb1 * pFalseMatchb1Val

  pFalseMatchSingleCell = np.sum(np.array(pFalseMatchb1ValDict.values()))

  return pFalseMatchPair, pFalseMatchSingleCell



def computeL2CellPairsFalseMatchConditionalProb(
        b1Val, b2Val, nVal, mVal, wVal, kVal, cVal):
  """
  Given that an L4 SDR with b1=10 bits overlap with L2 cell 1
  What is the chance that this SDR has b2=10 bits overlap with L2 cell 2?
  """

  n = Symbol("n", positive=True)
  m = Symbol("m", positive=True)
  w = Symbol("w", positive=True)
  k = Symbol("k", positive=True)
  c = Symbol("c", positive=True)
  b1 = Symbol("b1", positive=True)
  b2 = Symbol("b2", positive=True)

  numOverlap = n * m * (1 - pow(1 - (c * c) / (w * n * m), k))
  numCellsInUnion = n * m * (1 - pow(1 - c / (n * m), k))
  numTotal = binomial(n*m - numCellsInUnion, w-b1) * binomial(numCellsInUnion, b1)

  numOverlapVal = int(numOverlap.subs(k, kVal).subs(c, cVal).subs(n, nVal).
                      subs(m, mVal).subs(w, wVal).evalf())
  numCellsInUnionVal = int(numCellsInUnion.subs(k, kVal).subs(c, cVal).
                           subs(n, nVal).subs(m, mVal).evalf())

  numTotalVal = numTotal.subs(b1, b1Val).subs(b2, b2Val).\
    subs(n, nVal).subs(m, mVal).subs(k, kVal).subs(w, wVal).subs(c, cVal).evalf()

  minI = max(max(b1Val - (numCellsInUnionVal - numOverlapVal), 0),
             max(b2Val - (numCellsInUnionVal - numOverlapVal), 0))

  maxI = min(b1Val, b2Val)
  numMatchPair = 0
  for i in range(minI, maxI+1):
    numMatchPair += (binomial(numOverlapVal, i) *
                     binomial(numCellsInUnionVal - numOverlapVal, b2Val - i) *
                     binomial(numCellsInUnionVal - numOverlapVal, b1Val - i) *
                     binomial(nVal * mVal - 2 * numCellsInUnionVal + numOverlapVal,
                       wVal - b1Val - b2Val + i))

  pFalseMatchPair = numMatchPair / numTotalVal

  print "Match SDR # {} Total SDR # {} p={}".format(numMatchPair, numTotalVal, pFalseMatchPair)
  return pFalseMatchPair



def plotFalseMatchErrorSingleCell(cValList, thetaValList):
  """
  False Match error for single L2 cell
  :param cValList:
  """
  kValList = np.arange(0, 200, 5)

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
  ax[0].set_ylim([pow(10, -10), 1])
  ax[0].legend(legendList, loc=4)

  ax[1].set_xlabel('# (feature, location)')
  ax[1].set_ylabel('# connections')
  plt.tight_layout()



def runExperimentFalseMatchConditionalPairError():
  nVal = 150
  mVal = 16
  wVal = 10
  kVal = 40
  b1Val = 5
  b2Val = 5

  cValList = [5, 6, 7, 8, 9, 10]
  pFalseMatchPair = []
  for cVal in cValList:
    pFalseMatchPair.append(computeL2CellPairsFalseMatchConditionalProb(
      b1Val, b2Val, nVal, mVal, wVal, kVal, cVal))

  print "Verify Equation with simulations"
  pFalseMatchPairSimulate = []
  for cVal in cValList:
    b2Overlap = simulateL2CellPairsConditionalFalseMatch(
      b1Val, nVal, wVal, mVal, cVal, kVal)
    pFalseMatchPairSimulate.append(np.mean(b2Overlap==b2Val))

    print "b1 {} b2 {} c {} prob {}".format(
      b1Val, b2Val, cVal, pFalseMatchPairSimulate[-1])

  fig, ax = plt.subplots(1)
  ax.plot(cValList, pFalseMatchPair,'-o')
  ax.plot(cValList, pFalseMatchPairSimulate, '--rx')
  ax.set_ylabel("P(oj={}|oi={})".format(b1Val, b2Val))
  ax.set_xlabel("Connection # per SDR")
  plt.legend(['equation', 'simulation'])
  plt.savefig('ConditionalFalseMatchPairErrorVsC.pdf')



def runExperimentSingleVsPairMatchError():
  nVal = 150
  mVal = 16
  wVal = 10
  kVal = 40
  thetaValList = [5, 5, 5, 5, 5, 5]
  cValList = [5, 6, 7, 8, 9, 10]

  pFalseMatchPairSingle = []
  pFalseMatchPairList = []
  for i in range(len(cValList)):
    pFalseMatchPair, pFalseMatchSingleCell = computeL2CellPairsFalseMatchChance(
      thetaValList[i], nVal, mVal, wVal, kVal, cValList[i])

    print "c={} theta={} single error {} pair error {}".format(cValList[i],
                                                            thetaValList[i],
                                                            pFalseMatchSingleCell,
                                                            pFalseMatchPair)
    pFalseMatchPairSingle.append(pFalseMatchSingleCell)
    pFalseMatchPairList.append(pFalseMatchPair)

  fig, ax = plt.subplots(1)
  ax.semilogy(cValList, pFalseMatchPairSingle, '-bo')
  ax.semilogy(cValList, pFalseMatchPairList, '-go')
  ax.set_ylabel('False Match Error')
  ax.set_xlabel('# connections per pattern')
  plt.legend(['Single Output Neuron', 'Pair of Output Neurons'])
  plt.savefig('FalseMatchPairErrorVsC.pdf')



def runExperimentUnionSize():
  nVal = 150
  mVal = 16
  wVal = 10

  fig, ax = plt.subplots(1, 1)
  legendList = []
  for cVal in [5, 6, 7, 8]:
    # theoretical values
    numCellsVsK = []
    kValList = np.arange(1, 200, 20)
    for kVal in kValList:
      numCellsInUnionVal = calculateNumCellsVsK(kVal, nVal, cVal, mVal)
      numCellsVsK.append(numCellsInUnionVal)
    legendList.append("c={}".format(cVal))
    ax.plot(kValList, numCellsVsK)

  for cVal in  [5, 6, 7, 8]:
    # simulation values
    numCellsVsKsim = []
    kValListSparse = np.arange(1, 200, 20)
    for kVal in kValListSparse:
      numCellsInUnionValSim = simulateNumCellsVsK(kVal, nVal, cVal, mVal)
      numCellsVsKsim.append(numCellsInUnionValSim)
    ax.plot(kValListSparse, numCellsVsKsim, 'ko')

  ax.set_xlabel("# (feature, object) pair")
  ax.set_ylabel("# connected inputs per output cell")
  ax.legend(legendList, loc=2)
  plt.savefig('UnionSizeVsK.pdf')


if __name__ == "__main__":

  nVal = 150
  mVal = 16
  wVal = 10
  cVal = 5

  # plot the number of L4 cells that are connected to L2, as a function of
  # (feature locaiton) pairs per object
  runExperimentUnionSize()

  # plot the false match error for single L2 cell
  plotFalseMatchErrorSingleCell(cValList=[7, 7, 7, 7], thetaValList=[4, 5, 6, 7])
  plt.savefig('FalseMatchErrVsK_FixedCVaryingTheta.pdf')

  plotFalseMatchErrorSingleCell(cValList=[5, 6, 7, 8], thetaValList=[5, 5, 5, 5])
  plt.savefig('FalseMatchErrVsK_FixedThetaVaryingC.pdf')

  plotFalseMatchErrorSingleCell(cValList=[5, 10, 20, 30, 40], thetaValList=[3, 6, 12, 18, 24])
  plt.savefig('FalseMatchErrVsK_VaryingThetaandC.pdf')

  # plot conditional false match error
  # given that an L4 SDR with b1=10 bits overlap with L2 cell 1
  # what is the chance that this SDR has b2=10 bits overlap with L2 cell 2?
  runExperimentFalseMatchConditionalPairError()

  # plot simultaneous false match error for a pair of L2 cells
  # what is the chance that an L4 SDR falsely activate two L2 cells?
  runExperimentSingleVsPairMatchError()

  # Run simulaiton to get the distribution of output cells that are falsely
  # activated simultaneously
  simulateL4L2Pooling(cVal-1, nVal, wVal, mVal, cVal, k=100)

  calculateNumCellsVsK(100, nVal, cVal, mVal)
  computeL2CellPairsFalseMatchChance(cVal - 1, nVal, mVal, wVal, 100, cVal)
