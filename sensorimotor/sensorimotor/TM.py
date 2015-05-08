#! /usr/bin/env python
# ----------------------------------------------------------------------
#  Copyright (C) 2010, Numenta Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc. No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.
# ----------------------------------------------------------------------

import numpy
from numpy import *
import sys
import time
import cPickle as pickle
from itertools import product
import pprint
import copy

import nupic.math
from nupic.support.consoleprinter import ConsolePrinterMixin
from nupic.bindings.math import Random
from nupic.bindings.algorithms import isSegmentActive, getSegmentActivityLevel

# Default verbosity while running unit tests
VERBOSITY = 0

# The numpy equivalent to the floating point type used by NTA
dtype = nupic.math.GetNTAReal()

class TM(ConsolePrinterMixin):

  """
  Class implementing the temporal memory algorithm for sequence learning as described in the
  published Cortical Learning Algorithm documentation.  The implementation here
  attempts to closely match the pseudocode in the documentation. This
  implementation does contain several additional bells and whistles such as
  a column confidence measure.
  """

  ##############################################################################
  # todo: Have some higher level flags for fast learning, HiLo, Pooling, etc.
  def __init__(self,
               numberOfCols =500,
               cellsPerColumn =4,
               #columnsShape = # todo: need to specify topology at some point
               initialPerm =0.2, # todo: check perm numbers with Ron
               connectedPerm =0.8,
               newSynapseCount =10,
               permanenceInc =0.1,
               permanenceDec =0.05,
               permanenceMax =1.0,
               activationThreshold =8, # 3/4 of newSynapseCount TODO make fraction
               minThreshold =8,
               maxAge = 1,
               globalDecay=0.05,
               segUpdateValidDuration =5,
               burnIn =2,             # Used for evaluating the prediction score
               collectStats =False,    # If true, collect training and inference stats
               seed =42,
               verbosity =VERBOSITY,
               ):
    """
    Construct the TM

    @param numberOfCols Number of columns, should match the dimension of input

    @param cellsPerColumn Number of cells per column. Different cells in the same
                  column represents different context of the same input

    @param initialPerm initial permanance when a new synapse is added

    @param connectedPerm The threshold of permanance for connected synapse.

    @param newSynapseCount Maximal number of synapses for each segment.

    @param permanenceInc Incremants of permanance during learning, must be positive

    @param permanenceDec Decremants of permanance during learning, must be postive

    @param permanenceMax Maximal permanance. permamnance never exceeds this value

    @param activationThreshold The threshold to activate a segment. Number of activated
                  synapses must exceed this value to activate a dendritic segment

    @param minThreshold A cell becomes a learning cell if it has a segment with activity
                  level above minThreshold

    @param maxAge Controls global decay. Global decay will only decay segments
                  that have not been activated for maxAge iterations, and will
                  only do the global decay loop every maxAge iterations. The
                  default (maxAge=1) reverts to the behavior where global decay
                  is applied every iteration to every segment. Using maxAge > 1
                  can significantly speed up the TP when global decay is used.

    @param globalDecay The amount for global decay. For each segment that has not been a
                  good predictor in the last maxAge iterations. All the synapses
                  will be decreased by globalDecay

    @param burnIn Used for evaluating the prediction score. Default is 2.

    @param collectStats If True, collect training / inference stats. Default is
                        False.

    @param seed   seed for random number generator

    """

    ConsolePrinterMixin.__init__(self, verbosity)

    #---------------------------------------------------------------------------------
    # Seed random number generator
    if seed >= 0:
      self._random = Random(seed)
    else:
      self._random = Random(numpy.random.randint(256))

    #---------------------------------------------------------------------------------
    # Store creation parameters
    self.numberOfCols = numberOfCols
    self.cellsPerColumn = cellsPerColumn
    self._numberOfCells = numberOfCols * cellsPerColumn
    self.initialPerm = numpy.float32(initialPerm)
    self.connectedPerm = numpy.float32(connectedPerm)
    self.minThreshold = minThreshold
    self.newSynapseCount = newSynapseCount
    self.permanenceInc = numpy.float32(permanenceInc)
    self.permanenceDec = numpy.float32(permanenceDec)
    self.permanenceMax = numpy.float32(permanenceMax)
    self.globalDecay = numpy.float32(globalDecay)
    self.activationThreshold = activationThreshold
    self.segUpdateValidDuration = segUpdateValidDuration
    self.burnIn = burnIn
    self.collectStats = collectStats
    self.seed = seed
    self.verbosity = verbosity
    self.maxAge = maxAge

    self.segUpdateValidDuration = 1

    #---------------------------------------------------------------------------------
    # Create data structures
    self.activeColumns = [] # list of indices of active columns

    stateShape = (self.numberOfCols, self.cellsPerColumn)

    # Keep integers rather than bools. Float?
    self.activeState = {}
    self.activeState["t"] = numpy.zeros(stateShape, dtype="int8")
    self.activeState["t-1"] = numpy.zeros(stateShape, dtype="int8")

    self.learnState = {}
    self.learnState["t"] = numpy.zeros(stateShape, dtype="int8")
    self.learnState["t-1"] = numpy.zeros(stateShape, dtype="int8")

    self.predictedState = {}
    self.predictedState["t"] = numpy.zeros(stateShape, dtype="int8")
    self.predictedState["t-1"] = numpy.zeros(stateShape, dtype="int8")

    self.confidence = {}
    self.confidence["t"] = numpy.zeros(stateShape, dtype="float32")
    self.confidence["t-1"] = numpy.zeros(stateShape, dtype="float32")

    self.colConfidence = {}
    self.colConfidence["t"] = numpy.zeros(self.numberOfCols, dtype="float32")
    self.colConfidence["t-1"] = numpy.zeros(self.numberOfCols, dtype="float32")

    # Cells are indexed by column and index in the column
    # Every self.cells[column][index] contains a list of segments
    # Each segment is a structure of class Segment

    self.cells = []
    for c in xrange(self.numberOfCols):
      self.cells.append([])
      for i in xrange(self.cellsPerColumn):
        self.cells[c].append([])

    # iteration index
    self.iterationIdx = 0

    # iteration index in the learning mode
    self.lrnIterationIdx = 0

    # unique segment id, so we can put segments in hashes
    self.segID = 0

    self.currentOutput = None # for checkPrediction

    #---------------------------------------------------------------------------------
    # If True, the TP will compute a signature for each sequence
    self.collectSequenceStats = False

    self.resetCalled = False

    # All other members are ephemeral - don't need to be saved when we save
    #  state. So they get separated out into _initEphemerals, which also
    #  gets called when we are being restored from a saved state (via
    #  __setstate__)
    self._initEphemerals()

  ################################################################################
  def _getEphemeralMembers(self):
    """
    List of our member variables that we don't need to be saved
    """
    return [
      'segmentUpdates',
      '_internalStats',
      '_stats',
      ]

  #############################################################################
  def _initEphemerals(self):
    """
    Initialize all ephemeral members after being restored to a pickled state.
    """

    # We store the lists of segments updates, per cell, so that they can be applied
    # later during learning, when the cell gets bottom-up activation.
    # We store one list per cell. The lists are identified with a hash key which
    # is a tuple (column index, cell index).
    self.segmentUpdates = {}

    self.sequenceSignatures = []

    # Allocate and reset all stats
    self.resetStats()

  #############################################################################
  def __getstate__(self):
    """
    Return serializable state.  This function will return a version of the
    __dict__ with all "ephemeral" members stripped out.  "Ephemeral" members
    are defined as those that do not need to be (nor should be) stored
    in any kind of persistent file (e.g., NuPIC network XML file.)
    """
    # Make sure we access "cells" so we'll load it if needed...
    _ = self.cells

    state = self.__dict__.copy()

    for ephemeralMemberName in self._getEphemeralMembers():
      state.pop(ephemeralMemberName, None)

    state['_random'] = pickle.dumps(state['_random'])  # Must be done manually

    return state

  #############################################################################
  def __setstate__(self, state):
    """
    Set the state of ourself from a serialized state.
    """

    self.__dict__.update(state)
    self._random = pickle.loads(self._random)  # Must be done manually
    self._initEphemerals()


  ###########################################################################
  def __getattr__(self, name):
    """
    Patch __getattr__ so that we can catch the first access to 'cells' and load.

    This function is only called when we try to access an attribute that doesn't
    exist.  We purposely make sure that "self.cells" doesn't exist after
    unpickling so that we'll hit this, then we can load it on the first access.

    If this is called at any other time, it will raise an AttributeError.
    That's because:
    - If 'name' is "cells", after the first call, self._realCells won't exist
      so we'll get an implicit AttributeError.
    - If 'name' isn't "cells", I'd expect our super wouldn't have __getattr__,
      so we'll raise our own Attribute error.  If the super did get __getattr__,
      we'll just return what it gives us.
    """

    try:
      return super(TM, self).__getattr__(name)
    except AttributeError:
      raise AttributeError("'TM' object has no attribute '%s'" % name)

  #############################################################################
  def __del__(self):
    pass

  #############################################################################
  def setRandomSeed(self, seed):
    """ Seed the random number generator.
    This is used during unit testing to generate repeatable results.
    """
    self._random = Random(seed)

  #############################################################################
  def getRandomState(self):
    """ Return the random number state.
    This is used during unit testing to generate repeatable results.
    """
    return self._random.getstate()

  #############################################################################
  def setRandomState(self, state):
    """ Set the random number state.
    This is used during unit testing to generate repeatable results.
    """
    self._random.setstate(state)

  ################################################################################
  def reset(self,):
    """ Reset the state of all cells.
    This is normally used between sequences while training. All internal states
    are reset to 0.
    """

    self.activeState['t-1'].fill(0)
    self.activeState['t'].fill(0)
    self.predictedState['t-1'].fill(0)
    self.predictedState['t'].fill(0)
    self.learnState['t-1'].fill(0)
    self.learnState['t'].fill(0)
    self.confidence['t-1'].fill(0)
    self.confidence['t'].fill(0)

    # Flush the segment update queue
    self.segmentUpdates = {}

    self._internalStats['nInfersSinceReset'] = 0

    #To be removed
    self._internalStats['curPredictionScore'] = 0
    #New prediction score
    self._internalStats['curPredictionScore2']   = 0
    self._internalStats['curFalseNegativeScore'] = 0
    self._internalStats['curFalsePositiveScore'] = 0

    self._internalStats['curMissing'] = 0
    self._internalStats['curExtra'] = 0


    # When a reset occurs, set prevSequenceSignature to the signature of the
    # just-completed sequence and start accumulating histogram for the next
    # sequence.
    self._internalStats['prevSequenceSignature'] = None
    if self.collectSequenceStats:
      if self._internalStats['confHistogram'].sum() > 0:
        sig = self._internalStats['confHistogram'].copy()
        sig.reshape(self.numberOfCols * self.cellsPerColumn)
        self._internalStats['prevSequenceSignature'] = sig
      self._internalStats['confHistogram'].fill(0)

    self.resetCalled = True

  ################################################################################
  def resetStats(self):
    """ Reset the learning and inference stats. This will usually be called by
    user code at the start of each inference run (for a particular data set).
    """
    self._stats = dict()
    self._internalStats = dict()

    self._internalStats['nInfersSinceReset'] = 0
    self._internalStats['nPredictions'] = 0

    #New prediction score
    self._internalStats['curPredictionScore2']     = 0
    self._internalStats['predictionScoreTotal2']   = 0
    self._internalStats['curFalseNegativeScore']   = 0
    self._internalStats['falseNegativeScoreTotal'] = 0
    self._internalStats['curFalsePositiveScore']   = 0
    self._internalStats['falsePositiveScoreTotal'] = 0

    self._internalStats['pctExtraTotal'] = 0
    self._internalStats['pctMissingTotal'] = 0
    self._internalStats['curMissing'] = 0
    self._internalStats['curExtra'] = 0
    self._internalStats['totalMissing'] = 0
    self._internalStats['totalExtra'] = 0

    # Sequence signature statistics. Note that we don't reset the sequence
    # signature list itself.
    self._internalStats['prevSequenceSignature'] = None
    if self.collectSequenceStats:
      self._internalStats['confHistogram'] = \
          numpy.zeros((self.numberOfCols, self.cellsPerColumn), dtype="float32")


  ################################################################################
  def getStats(self):
    """ Return the current learning and inference stats. This returns a dict
    containing all the learning and inference stats we have collected since the
    last resetStats(). If collectStats is False, then None is returned.

    The following keys are returned in the dict when collectStats is True:

      nPredictions:           the number of predictions. This is the total number
                              of inferences excluding burn-in and the last
                              inference.

      curPredictionScore:     the score for predicting the current input (predicted
                              during the previous inference)

      curMissing:             the number of bits in the current input that were
                              not predicted to be on.

      curExtra:               the number of bits in the predicted output that are
                              not in the next input

      predictionScoreTotal:   the sum of every prediction score to date

      predictionScoreAvg:     predictionScoreTotal / nPredictions

      pctMissingTotal:        the total number of bits that were missed over all
                              predictions

      pctMissingAvg:          pctMissingTotal / nPredictions

      prevSequenceSignature:  signature for the sequence immediately preceding the
                              last reset. 'None' if collectSequenceStats is False

    """

    if not self.collectStats:
      return None

    self._stats['nPredictions'] = self._internalStats['nPredictions']
    self._stats['curMissing'] = self._internalStats['curMissing']
    self._stats['curExtra'] = self._internalStats['curExtra']
    self._stats['totalMissing'] = self._internalStats['totalMissing']
    self._stats['totalExtra'] = self._internalStats['totalExtra']

    nPredictions = max(1, self._stats['nPredictions'])

    #New prediction score
    self._stats['curPredictionScore2'] = self._internalStats['curPredictionScore2']
    self._stats['predictionScoreAvg2'] = self._internalStats['predictionScoreTotal2'] \
                                         / nPredictions
    self._stats['curFalseNegativeScore'] = self._internalStats['curFalseNegativeScore']
    self._stats['falseNegativeAvg'] = self._internalStats['falseNegativeScoreTotal'] \
                                      / nPredictions
    self._stats['curFalsePositiveScore'] = self._internalStats['curFalsePositiveScore']
    self._stats['falsePositiveAvg'] = self._internalStats['falsePositiveScoreTotal'] \
                                      / nPredictions

    self._stats['pctExtraAvg'] = self._internalStats['pctExtraTotal'] \
                                   / nPredictions
    self._stats['pctMissingAvg'] = self._internalStats['pctMissingTotal'] \
                                   / nPredictions

    # This will be None if collectSequenceStats is False
    self._stats['prevSequenceSignature'] = self._internalStats['prevSequenceSignature']

    return self._stats


  ################################################################################
  def _updateStatsInferEnd(self, stats, bottomUpNZ, predictedState, confidence):
    """ Called at the end of learning and inference, this routine will update
    a number of stats in our _internalStats dictionary, including:
      1. Our computed prediction score
      2. ...

    Parameters:
    ------------------------------------------------------------------
    bottomUpNZ: list of the active bottom-up inputs

    """

    # Return if not collecting stats
    if not self.collectStats:
      return
    stats['nInfersSinceReset'] += 1


    # Compute the prediction score, how well the prediction from the last
    #  time step predicted the current bottom-up input
    patternsToCheck = [bottomUpNZ]
    (numExtra2, numMissing2, confidences2) = self.checkPrediction2(
                                  patternNZs = [bottomUpNZ],
                                  output = predictedState,
                                  confidence = confidence)
    predictionScore, positivePredictionScore, negativePredictionScore = \
                                                          confidences2[0]

    # Store the stats that don't depend on burn-in
    stats['curPredictionScore2'] = float(predictionScore)
    stats['curFalseNegativeScore'] = 1.0 - float(positivePredictionScore)
    stats['curFalsePositiveScore'] = float(negativePredictionScore)

    stats['curMissing'] = numMissing2
    stats['curExtra'] = numExtra2


    # ----------------------------------------------------------------------
    # If we are passed the burn-in period, update the accumulated stats
    # Here's what various burn-in values mean:
    #   0: try to predict the first element of each sequence and all subsequent
    #   1: try to predict the second element of each sequence and all subsequent
    #   etc.
    if stats['nInfersSinceReset'] <= self.burnIn:
      return

    # Burn-in related stats
    stats['nPredictions'] += 1
    numExpected = max(1.0, float(len(bottomUpNZ)))

    stats['totalMissing'] += numMissing2
    stats['totalExtra'] += numExtra2
    stats['pctExtraTotal'] += 100.0 * numExtra2 / numExpected
    stats['pctMissingTotal'] += 100.0 * numMissing2 / numExpected
    stats['predictionScoreTotal2'] += float(predictionScore)
    stats['falseNegativeScoreTotal'] += 1.0 - float(positivePredictionScore)
    stats['falsePositiveScoreTotal'] += float(negativePredictionScore)

    if self.collectSequenceStats:
      # Collect cell confidences for every cell that correctly predicted current
      # bottom up input. Normalize confidence across each column
      cc = self.confidence['t-1'] * self.activeState['t']
      sconf = cc.sum(axis=1)
      for c in range(self.numberOfCols):
        if sconf[c] > 0:
          cc[c,:] /= sconf[c]

      # Update cell confidence histogram: add column-normalized confidence
      # scores to the histogram
      self._internalStats['confHistogram'] += cc


  ################################################################################
  # The following print functions for debugging.
  ################################################################################

  def printState(self, aState):
    """Print an integer array that is the same shape as activeState."""
    def formatRow(var, i):
      s = ''
      for c in range(self.numberOfCols):
        if c > 0 and c % 10 == 0:
          s += ' '
        s += str(var[c,i])
      s += ' '
      return s

    for i in xrange(self.cellsPerColumn):
      print formatRow(aState,i)

  def printConfidence(self, aState, maxCols = 20):
    """Print a floating point array that is the same shape as activeState."""
    def formatFPRow(var, i):
      s = ''
      for c in range(min(maxCols,self.numberOfCols)):
        if c > 0 and c % 10 == 0:
          s += '   '
        s += ' %5.3f'%var[c,i]
      s += ' '
      return s

    for i in xrange(self.cellsPerColumn):
      print formatFPRow(aState,i)

  def printColConfidence(self, aState, maxCols = 20):
    """Print up to maxCols number from a flat floating point array."""
    def formatFPRow(var):
      s = ''
      for c in range(min(maxCols,self.numberOfCols)):
        if c > 0 and c % 10 == 0:
          s += '   '
        s += ' %5.3f'%var[c]
      s += ' '
      return s

    print formatFPRow(aState)

  def printStates(self, printPrevious = True, printLearnState = True):

    nSpaces = 2 * self.numberOfCols - 3

    def formatRow(var, i):
      s = ''
      for c in range(self.numberOfCols):
        if c > 0 and c % 10 == 0:
          s += ' '
        s += str(var[c,i])
      s += ' '
      return s

    print "\nActive state"
    for i in xrange(self.cellsPerColumn):
      if printPrevious:
        print formatRow(self.activeState['t-1'], i),
      print formatRow(self.activeState['t'],i)

    print "Predicted state"
    for i in xrange(self.cellsPerColumn):
      if printPrevious:
        print formatRow(self.predictedState['t-1'], i),
      print formatRow(self.predictedState['t'],i)

    if printLearnState:
      print "Learn state"
      for i in xrange(self.cellsPerColumn):
        if printPrevious:
          print formatRow(self.learnState['t-1'], i),
        print formatRow(self.learnState['t'],i)

  def printOutput(self, y):
    print "Output"
    for i in xrange(self.cellsPerColumn):
      for c in xrange(self.numberOfCols):
        print int(y[c,i]),
      print

  def printInput(self, x):
    print "Input"
    for c in xrange(self.numberOfCols):
      print int(x[c]),
    print

  def printParameters(self):
    """Print the parameter settings for the TM."""
    print "numberOfCols=",self.numberOfCols
    print "cellsPerColumn=",self.cellsPerColumn
    print "minThreshold=",self.minThreshold
    print "newSynapseCount=",self.newSynapseCount
    print "activationThreshold=",self.activationThreshold
    print
    print "initialPerm=",self.initialPerm
    print "connectedPerm=",self.connectedPerm
    print "permanenceInc=", self.permanenceInc
    print "permanenceDec=", self.permanenceDec
    print "permanenceMax=", self.permanenceMax
    print "globalDecay=", self.globalDecay
    print
    print "segUpdateValidDuration=",self.segUpdateValidDuration


  def printActiveIndices(self, state, andValues=False):
    """Print the list of [column, cellIdx] indices for each of the active
    cells in state. """
    (cols, cellIdxs) = state.nonzero()

    if len(cols) == 0:
      print "NONE"
      return

    prevCol = -1
    for (col,cellIdx) in zip(cols, cellIdxs):
      if col != prevCol:
        if prevCol != -1:
          print "] ",
        print "Col %d: [" % (col),
        prevCol = col

      if andValues:
        print "%d: %s," % (cellIdx, state[col,cellIdx]),
      else:
        print "%d," % (cellIdx),
    print "]"

  def printComputeEnd(self, output, learn=False):
    """ Called at the end of inference to print out various diagnostic
    information based on the current verbosity level.
    """
    if self.verbosity >= 3:
      print "----- computeEnd summary: "
      print "numBurstingCols: %s, " % (self.activeState['t'].min(axis=1).sum()),
      print "curPredScore2: %s, " % (self._internalStats['curPredictionScore2']),
      print "curFalsePosScore: %s, " % (self._internalStats['curFalsePositiveScore']),
      print "1-curFalseNegScore: %s, " % (1-self._internalStats['curFalseNegativeScore']),
      print "numPredictedCells[t-1]: %s" % (self.predictedState['t-1'].sum()),
      print "numSegments: ",self.getNumSegments()

      print "----- activeState (%d on) ------" \
              % (self.activeState['t'].sum())
      self.printActiveIndices(self.activeState['t'])
      if self.verbosity >= 5:
        self.printState(self.activeState['t'])

      print "----- predictedState (%d on)-----" \
              % (self.predictedState['t'].sum())
      self.printActiveIndices(self.predictedState['t'])
      if self.verbosity >= 5:
        self.printState(self.predictedState['t'])

      print "----- cell confidence -----"
      self.printActiveIndices(self.confidence['t'], andValues=True)
      if self.verbosity >= 5:
        self.printConfidence(self.confidence['t'])

      print "----- confidence[t-1] for currently active cells -----"
      cc = self.confidence['t-1'] * self.activeState['t']
      self.printActiveIndices(cc, andValues=True)

      if self.verbosity == 4:
        print "Cells, predicted segments only:"
        self.printCells(predictedOnly=True)
      elif self.verbosity >= 5:
        print "Cells, all segments:"
        self.printCells(predictedOnly=False)
      print


  ################################################################################
  def printSegmentUpdates(self):
    print "=== SEGMENT UPDATES ===, Num = ",len(self.segmentUpdates)
    for key, updateList in self.segmentUpdates.iteritems():
      c,i = key[0], key[1]
      print c,i,updateList


  ################################################################################
  def printCell(self, c, i, onlyActiveSegments=False):

    if len(self.cells[c][i]) > 0:
      print "Column", c, "Cell", i, ":",
      print len(self.cells[c][i]), "segment(s)"
      for j,s in enumerate(self.cells[c][i]):
        isActive = self.isSegmentActive(s, self.activeState['t'])
        if not onlyActiveSegments or isActive:
          isActiveStr = "*" if isActive else " "
          print "  %sSeg #%-3d" % (isActiveStr, j),
          s.printSegment()

  ################################################################################
  def printCells(self, predictedOnly=False):

    if predictedOnly:
      print "--- PREDICTED CELLS ---"
    else:
      print "--- ALL CELLS ---"
    print "Activation threshold=", self.activationThreshold,
    print "min threshold=", self.minThreshold,
    print "connected perm=", self.connectedPerm

    for c in xrange(self.numberOfCols):
      for i in xrange(self.cellsPerColumn):
        if not predictedOnly or self.predictedState['t'][c,i]:
          self.printCell(c, i, predictedOnly)


  #############################################################################
  def getNumSegmentsInCell(self, c, i):
    """ Return the total number of synapses in cell (c,i)
    """
    return len(self.cells[c][i])

  #############################################################################
  def getNumSynapses(self):
    """ Return the total number of synapses
    """

    nSyns = self.getSegmentInfo()[1]
    return nSyns

  #############################################################################
  def getNumStrongSynapses(self):
    """ Return the total number of strong synapses
    """
    #todo: implement this, it is used by the node's getParameter() call
    return 0

  #############################################################################
  def getNumStrongSynapsesPerTimeSlot(self):
    """ Return the total number of strong synapses per time slot
    """
    #todo: implement this, it is used by the node's getParameter() call
    return 0

  #############################################################################
  def getNumSynapsesPerSegmentMax(self):
    """ Return the max # of synapses seen in any one segment
    """
    #todo: implement this, it is used by the node's getParameter() call
    return 0

  #############################################################################
  def getNumSynapsesPerSegmentAvg(self):
    """ Return the average number of synapses per segment
    """

    return float(self.getNumSynapses()) / max(1, self.getNumSegments())

  #############################################################################
  def getNumSegments(self):
    """ Return the total number of segments
    """
    nSegs = self.getSegmentInfo()[0]
    return nSegs

  #############################################################################
  def getNumCells(self):
    """ Return the total number of cells
    """
    return self.numberOfCols * self.cellsPerColumn

  ################################################################################
  def getSegmentOnCell(self, c, i, segIdx):
    """Return the segment on cell (c,i) with index sidx.
    Returns the segment as following list:
      [  [segmentID, sequenceSegmentFlag, frequency],
         [col1, idx1, perm1],
         [col2, idx2, perm2], ...
      ]
    """
    return self.cells[c][i][segIdx]

  #############################################################################
  class SegmentUpdate():
    """
    Class used to carry instructions for updating a segment.
    """

    def __init__(self, c, i, seg=None, activeSynapses=[]):
      """
          Construct the SegmentUpdate class
          @param c: column index

          @param i: cell index in the column

          @param seg: The segment object itself, not an index (can be None)

          @param activeSynapses: active synapses due to lateral connections

      """
      self.columnIdx = c
      self.cellIdx = i
      self.segment = seg # The segment object itself, not an index (can be None)
      self.activeSynapses = activeSynapses
      self.sequenceSegment = False
      self.phase1Flag = False

    # Just for debugging
    def __str__(self):

      return "Seg update: cell=[%d,%d]" % (self.columnIdx, self.cellIdx)  \
             + ", seg=" + str(self.segment) \
             + ", synapses=" + str(self.activeSynapses) \
             + ", seq seg=" + str(self.sequenceSegment) \
             + ", phase1=" + str(self.phase1Flag)

  ################################################################################
  def addToSegmentUpdates(self, c, i, segUpdate):
    """
    Store a dated potential segment update. The "date" (iteration index) is used
    later to determine whether the update is too old and should be forgotten.
    This is controlled by parameter segUpdateValidDuration.
    """

    # Sometimes we might be passed an empty update
    if segUpdate is None or len(segUpdate.activeSynapses) == 0:
      return

    key = (c,i) # key = (column index, cell index in column)

    # todo: scan list of updates for that cell and consolidate?
    # But watch out for dates!
    if self.segmentUpdates.has_key(key):
      self.segmentUpdates[key] += [(self.lrnIterationIdx, segUpdate)]
    else:
      self.segmentUpdates[key] = [(self.lrnIterationIdx, segUpdate)]


  ################################################################################
  def removeSegmentUpdate(self, updateInfo):
    """Remove a segment update (called when seg update expires or is processed)

    Parameters:
    --------------------------------------------------------------
    updateInfo:     (creationDate, SegmentUpdate)
    """

    # An updateInfo contains (creationDate, SegmentUpdate)
    (creationDate, segUpdate) = updateInfo

    # Key is stored in segUpdate itself...
    key = (segUpdate.columnIdx, segUpdate.cellIdx)

    self.segmentUpdates[key].remove(updateInfo)

  #############################################################################
  def computeOutput(self):

    """Computes output for both learning and inference. In both cases, the
    output is the boolean OR of activeState and predictedState at t.
    Stores currentOutput for checkPrediction."""

    # todo: This operation can be sped up by:
    #  1.)  Pre-allocating space for the currentOutput
    #  2.)  Making predictedState and activeState of type 'float32' up front
    #  3.)  Using logical_or(self.predictedState['t'], self.activeState['t'],
    #          self.currentOutput)
    #
    self.currentOutput = \
        logical_or(self.predictedState['t'], self.activeState['t'])

    return self.currentOutput.reshape(-1).astype('float32')

  #############################################################################
  def getActiveState(self):
    """ Return the current active state. This is called by the node to
    obtain the sequence output of the TP.

    """

    # todo: This operation can be sped up by making  activeState of
    #         type 'float32' up front.
    return self.activeState['t'].reshape(-1).astype('float32')


  def getPredictedState(self):
    """
    Return a numpy array, predictedCells, representing the current predicted
    state.

    predictedCells[c][i] represents the state of the i'th cell in the c'th
    column.

    @returns numpy array of predicted cells, representing the current predicted
    state. predictedCells[c][i] represents the state of the i'th cell in the c'th
    column.
    """
    return self.predictedState['t']

  #############################################################################
  def predict(self, nSteps):
    """
    This function gives the future predictions for <nSteps> timesteps starting
    from the current TP state. The TP is returned to its original state at the
    end before returning.

    1) We save the TP state.
    2) Loop for nSteps
          a) Turn-on with lateral support from the current active cells
          b) Set the predicted cells as the next step's active cells. This step
             in learn and infer methods use input here to correct the predictions.
             We don't use any input here.
    3) Revert back the TP state to the time before prediction

    Parameters:
    --------------------------------------------
    nSteps:      The number of future time steps to be predicted
    retval:      all the future predictions - a numpy array of type "float32" and
                 shape (nSteps, numberOfCols).
                 The ith row gives the tp prediction for each column at
                 a future timestep (t+i+1).

    """

    # Save the TP dynamic state, we will use to revert back in the end
    pristineTPDynamicState = self._getTPDynamicState()

    assert (nSteps>0)

    # multiStepColumnPredictions holds all the future prediction.
    multiStepColumnPredictions = numpy.zeros((nSteps, self.numberOfCols),
                                             dtype="float32")

    # This is a (nSteps-1)+half loop. Phase 2 in both learn and infer methods
    # already predicts for timestep (t+1). We use that prediction for free and save
    # the half-a-loop of work.

    step = 0
    while True:

      # We get the prediction for the columns in the next time step from
      # the topDownCompute method. It internally uses confidences.
      multiStepColumnPredictions[step,:] = self.topDownCompute()

      # Cleanest way in python to handle one and half loops
      if step == nSteps-1:
        break
      step += 1

      # Copy t-1 into t
      self.activeState['t-1'][:,:] = self.activeState['t'][:,:]
      self.predictedState['t-1'][:,:] = self.predictedState['t'][:,:]
      self.confidence['t-1'][:,:] = self.confidence['t'][:,:]

      # Predicted state at "t-1" becomes the active state at "t"
      self.activeState['t'][:,:] = self.predictedState['t-1'][:,:]

      # Predicted state and confidence are set in phase2.
      self.predictedState['t'].fill(0)
      self.confidence['t'].fill(0.0)
      self.computePhase2(doLearn=False)

    # Revert the dynamic state to the saved state
    self._setTPDynamicState(pristineTPDynamicState)


    return multiStepColumnPredictions

  #############################################################################
  def _getTPDynamicStateVariableNames(self,):
    """
    Any newly added dynamic states in the TP should be added to this list.

    Parameters:
    --------------------------------------------
    retval:       The list of names of TP dynamic state variables.

    """
    return ["activeState",
            "learnState",
            "predictedState",
            "confidence"]

  #############################################################################
  def _getTPDynamicState(self,):
    """
    Parameters:
    --------------------------------------------
    retval:       A dict with all the dynamic state variable names as keys and
                  their values at this instant as values.
    """
    tpDynamicState = dict()
    for variableName in self._getTPDynamicStateVariableNames():
      tpDynamicState[variableName] = copy.deepcopy(self.__dict__[variableName])
    return tpDynamicState

  #############################################################################
  def _setTPDynamicState(self, tpDynamicState):
    """
    Set all the dynamic state variables from the <tpDynamicState> dict.

    <tpDynamicState> dict has all the dynamic state variable names as keys and
    their values at this instant as values.

    We set the dynamic state variables in the tp object with these items.

    """
    for variableName in self._getTPDynamicStateVariableNames():
      self.__dict__[variableName] = tpDynamicState.pop(variableName)


  #############################################################################
  def computePhase2(self, doLearn=False):
    """
    This is the phase 2 of learning, inference and multistep prediction. During
    this phase, all the cell with lateral support have their predictedState
    turned on and the firing segments are queued up for updates.

    Parameters:
    --------------------------------------------
    doLearn:      Boolean flag to queue segment updates during learning
    retval:       ?
    """

    # Phase 2: compute predicted state for each cell
    # - if a segment has enough horizontal connections firing because of
    #   bottomUpInput, it's set to be predicting, and we queue up the segment
    #   for reinforcement,
    # - if pooling is on, try to find the best weakly activated segment to
    #   reinforce it, else create a new pooling segment.
    for c in xrange(self.numberOfCols):

      buPredicted = False # whether any cell in the column is predicted
      for i in xrange(self.cellsPerColumn):
        # Iterate over each of the segments of this cell
        maxConfidence = 0
        for s in self.cells[c][i]:

          # sum(connected synapses) >= activationThreshold?
          if self.isSegmentActive(s, self.activeState['t']):

            self.predictedState['t'][c,i] = 1
            buPredicted = True
            maxConfidence = max(maxConfidence, s.dutyCycle(readOnly=True))

            if doLearn:
              s.totalActivations += 1    # increment activationFrequency
              s.lastActiveIteration = self.iterationIdx
              # mark this segment for learning
              activeUpdate = self.getSegmentActiveSynapses(c,i,s,'t')
              activeUpdate.phase1Flag = False
              self.addToSegmentUpdates(c, i, activeUpdate)

        # Store the max confidence seen among all the weak and strong segments
        #  as the cell's confidence.
        self.confidence['t'][c,i] = maxConfidence


  def compute(self, bottomUpInput, enableLearn, computeInfOutput=None):
    """Computes output for both learning and inference. In both cases, the
    output is the boolean OR of activeState and predictedState at t.
    Stores currentOutput for checkPrediction."""

    if enableLearn:
      self.learn(bottomUpInput)
    else:
      self.infer(bottomUpInput)

  #############################################################################
  # During inference the code is split into two phases that occur one after
  # another:
  #
  # Phase 1: computing the current state for each cell, and,
  # Phase 2: computing the predicted state for each cell.
  #
  # Upon exiting, activeState and predictedState will be set for each cell.
  # The boolean OR of activeState and predictedState forms the temporal pooler
  # output and is sent to the next level.
  #
  # Phase 1: for each winning column (determined by the SP), we decide whether
  # to burst or not. If the column was predicted by some cell, then we burst
  # only those cells (possibly several). If it wasn't predicted by any cell,
  # we burst the whole column.
  #
  # Phase 2: all cells that have at least one segment active turn their predicted
  # output on.
  #
  # todo: keep lists of active segments so that we don't have to recompute them
  # twice (at t-1 and then at t again)
  #
  ################################################################################
  def infer(self, bottomUpInput):
    """
    Parameters:
    --------------------------------------------
    input:        Current bottom-up input, dense
    retval:       ?
    """

    self.iterationIdx = self.iterationIdx + 1

    if self.verbosity >= 3:
      print "\n==== Iteration: %d =====" % (self.iterationIdx)
      print "Active cols:", bottomUpInput.nonzero()[0]

    # Copy t into t-1
    # Don't need to copy learnState, which is not used in inference
    self.activeState['t-1'][:,:] = self.activeState['t'][:,:]
    self.predictedState['t-1'][:,:] = self.predictedState['t'][:,:]
    self.confidence['t-1'][:,:] = self.confidence['t'][:,:]
    self.activeState['t'].fill(0)
    self.predictedState['t'].fill(0)
    self.confidence['t'].fill(0.0)

    # Phase 1: calculate current state for each cell
    # For each column (winning in the SP):
    # - if the bottom up input was predicted by one of the sequence
    #   segments of a cell, this cell bursts,
    # - if the bottom up input was not predicted by any cell, the entire
    #   column bursts.
    activeColumns = bottomUpInput.nonzero()[0]

    for c in activeColumns:

      buPredicted = False
      predictingCells = numpy.where(self.predictedState['t-1'][c] == 1)[0]

      for i in predictingCells:
        buPredicted = True
        self.activeState['t'][c,i] = 1

      if not buPredicted:
        self.activeState['t'][c,:] = 1 # whole column bursts

    # Phase 2: calculate predictive output for each cell
    # A cell turns on its predictedState if any of its segments
    # has enough horizontal connections currently firing due to
    # bottom up input.
    self.computePhase2(doLearn=False)

    # Update the prediction score stats
    if self.collectStats:
      # import pdb; pdb.set_trace()
      self._updateStatsInferEnd(self._internalStats,
                                activeColumns,
                                self.predictedState['t-1'],
                                self.confidence['t-1'])



    # Get the output
    output = self.computeOutput()

    # Print diagnostic information based on the current verbosity level
    self.printComputeEnd(output, learn=False)

    self.resetCalled = False

    return output

  #############################################################################
  # The learning algorithm can be summarized in the following three steps:
  #
  # 1) If a column wins, choose one or more cells to fire
  #
  # 2) If a cell fires (for any reason):
  # 2.a) Queue up a reinforcement for the predicting segment, i.e the segment
  # that became active based on current SP outputs, and,
  # 2.b) Queue up reinforcement for the segment that best matches output from
  # the previous SP outputs.
  #
  # 3) If the cell bursts (the activeState output is on in the current time step),
  # then implement all the queued up changes. Otherwise, if the cell does not fire
  # at all, then throw away all the queued up changes. In this case we negatively
  # reinforce all segments that were marked as predicting segments.
  ################################################################################
  def learn(self, bottomUpInput):
    """
    Parameters:
    --------------------------------------------
    input:        Current bottom-up input
    retval:       ?
    """

    self.lrnIterationIdx = self.lrnIterationIdx + 1
    self.iterationIdx = self.iterationIdx + 1
    if self.verbosity >= 3:
      print "\n==== Iteration: %d =====" % (self.iterationIdx)
      print "Active cols:", bottomUpInput.nonzero()[0]


    if self.verbosity >= 4:
      print len(self.segmentUpdates), "updates"
      for k,v in self.segmentUpdates.iteritems():
        print 'cell:', k[0] * self.cellsPerColumn + k[1],
        for vv in v:
          print 'seg:', vv[1].segment,
          print 'timeStamp:', vv[0],
          print '/ src cells:', vv[1].activeSynapses

    # Copy t into t-1
    # This time also copy over learnState.
    self.activeState['t-1'][:,:] = self.activeState['t'][:,:]
    self.activeState['t'].fill(0)
    self.predictedState['t-1'][:,:] = self.predictedState['t'][:,:]
    self.predictedState['t'].fill(0)
    self.learnState['t-1'][:,:] = self.learnState['t'][:,:]
    self.learnState['t'][:,:] = 0
    self.confidence['t-1'][:,:] = self.confidence['t'][:,:]
    self.confidence['t'].fill(0.0)

    # Update segment duty cycles if we are crossing a "tier"
    # We determine if it's time to update the segment duty cycles. Since the
    # duty cycle calculation is a moving average based on a tiered alpha, it is
    # important that we update all segments on each tier boundary
    if self.lrnIterationIdx in Segment.dutyCycleTiers:
      for c, i in product(xrange(self.numberOfCols),
                                    xrange(self.cellsPerColumn)):
        for segment in self.cells[c][i]:
          segment.dutyCycle()


    # Phase 1: compute current state for each cell
    # For each column (winning in the SP):
    # - if the bottom up input was predicted by one of the sequence
    #   segments of a cell, this cell bursts, (those cells also become
    #   learning cells),
    # - if the bottom up input was not predicted by any cell, the entire
    #   column bursts, AND:
    #   - if a cell had a sequence segment above minThreshold, it becomes
    #     a learning cell, else
    #   - we create a new sequence segment for that cell, and it also
    #     becomes a learning cell.
    activeColumns = bottomUpInput.nonzero()[0]
    numUnpredictedColumns = 0


    for c in activeColumns:

      # todo: cache this list when building it in iteration at t
      buPredicted = False  # Was this bottom up input predicted?
      predictingCells = numpy.where(self.predictedState['t-1'][c] == 1)[0]

      for i in predictingCells:
        # Convert from Numpy integer structures, for debugging in Komodo
        c,i = int(c), int(i)
        buPredicted = True
        self.activeState['t'][c,i] = 1
        self.learnState['t'][c,i] = 1

      # Turn on the active state of the whole column if the bottom up input was
      # not predicted
      if not buPredicted:
        numUnpredictedColumns += 1

        self.activeState['t'][c,:] = 1

        # We didn't find a cell to learn on (since no cell was predicted), so
        # just pick the best matching cell, or, failing that, the least allocated
        # cell.
        # sum(all synapses) >= minThreshold, "weak" activation
        i,s = self.getBestMatchingCell(c,self.activeState['t-1'])

        if s is not None:
          s.totalActivations += 1      # activationFrequency
          s.lastActiveIteration = self.iterationIdx
        else:
          # if best matching cell does not exist, then get least used cell
          i = self.getLeastUsedCell(c)
          s = None



          # Update a possibly weakly matching segment
          # todo: In here, should we only use learnState to check for
          # a weak match?
        self.learnState['t'][c,i] = 1
        self.activeState['t'][c,i] = 1    # In case we are in PAM mode

        # propose a list of synapse change
        segUpdate = self.getSegmentActiveSynapses(c,i,s,'t-1',
                            newSynapses = True)
        segUpdate.phase1Flag = True
        self.addToSegmentUpdates(c, i, segUpdate)

    # ----------------------------------------------------------------------
    # Phase 2: compute predicted state for each cell
    # - if a segment has enough horizontal connections firing because of
    #   bottomUpInput, it's set to be predicting, and we queue up the segment
    #   for reinforcement,
    self.computePhase2(doLearn=True)

    # ----------------------------------------------------------------------
    # Phase 3: update synapses for currently active cells (due to bottom up input)
    # that have queued up segment updates, or cells that just stopped predicting
    # (those cells that stopped predicting are _negatively_ reinforced).
    # Also clean up queues by removing seg updates that are too old.
    self.processSegmentUpdates()

    # ----------------------------------------------------------------------
    # Phase 4: Apply global decay, and remove synapses and/or segments.
    # Synapses are removed if their permanence value is <= 0.
    # Segments are removed when they don't have synapses anymore.
    # Removal of synapses can trigger removal of whole segments!
    # todo: isolate the synapse/segment retraction logic so that
    # it can be called in adaptSegments, in the case where we
    # do global decay only episodically.
    if self.globalDecay > 0.0 and ((self.iterationIdx % self.maxAge) == 0):
      for c, i in product(xrange(self.numberOfCols), xrange(self.cellsPerColumn)):

        segsToDel = [] # collect and remove outside the loop
        for segment in self.cells[c][i]:
          age = self.iterationIdx - segment[0][self.kSegLastActiveIteration]
          if age <= self.maxAge:
            continue

          #print "Decrementing seg age %d:" % (age), c, i, segment
          synsToDel = [] # collect and remove outside the loop
          for synapse in segment.syns: # skip sequenceSegment flag

            synapse[2] = synapse[2] - self.globalDecay # decrease permanence

            if synapse[2] <= 0:
              synsToDel.append(synapse) # add to list to delete

          if len(synsToDel) == len(segment.syns): # 1 for sequenceSegment flag
            segsToDel.append(segment) # will remove the whole segment
          elif len(synsToDel) > 0:
            for syn in synsToDel: # remove some synapses on segment
              segment.syns.remove(syn)

        for seg in segsToDel: # remove some segments of this cell
          self.cleanUpdatesList(c,i,seg)
          self.cells[c][i].remove(seg)


    # Update the prediction score stats
    # Learning always includes inference
    if self.collectStats:
      self._updateStatsInferEnd(self._internalStats, activeColumns,
                                self.predictedState['t-1'],
                                self.confidence['t-1'])

    # Finally return learning output
    output = self.computeOutput()

    # Print diagnostic information based on the current verbosity level
    self.printComputeEnd(output, learn=True)

    self.resetCalled = False

    return self.computeOutput()

  #############################################################################
  def columnConfidences(self, cellConfidences=None):
    """ Compute the column confidences given the cell confidences. If
    None is passed in for cellConfidences, it uses the stored cell confidences
    from the last compute.

    Parameters:
    ----------------------------
    cellConfidencs      : cell confidences to use, or None to use the
                          the current cell confidences.

    retval:             : Column confidence scores.
    """

    if cellConfidences is None:
      cellConfidences = self.confidence['t']

    colConfidences = cellConfidences.sum(axis=1)

    # Make the max column confidence 1.0
    #colConfidences /= colConfidences.max()

    return colConfidences


  #############################################################################
  def topDownCompute(self, topDownIn=None):
    """ Top-down compute - generate expected input given output of the TP

    Parameters:
    ----------------------------
    topDownIn           : top down input from the level above us

    retval:             : best estimate of the TP input that would have generated
                          bottomUpOut.
    """

    # For now, we will assume there is no one above us and that bottomUpOut is
    #  simply the output that corresponds to our currently stored column
    # confidences.

    # Simply return the column confidences
    return self.columnConfidences()


  ################################################################################
  def trimSegmentsInCell(self, colIdx, cellIdx, segList, minPermanence,
              minNumSyns):
    """ This method goes through a list of segments for a given cell and
    deletes all synapses whose permanence is less than minPermanence and deletes
    any segments that have less than minNumSyns synapses remaining.

    Parameters:
    --------------------------------------------------------------
    colIdx:             Column index
    cellIdx:            cell index within the column
    segList:            List of segment references
    minPermanence:      Any syn whose permamence is 0 or < minPermanence will
                        be deleted.
    minNumSyns:         Any segment with less than minNumSyns synapses remaining
                        in it will be deleted.
    retval:             (numSegsRemoved, numSynsRemoved)
    """

    # Fill in defaults
    if minPermanence is None:
      minPermanence = self.connectedPerm
    if minNumSyns is None:
      minNumSyns = self.activationThreshold

    # Loop through all segments
    nSegsRemoved, nSynsRemoved = 0, 0
    segsToDel = [] # collect and remove segments outside the loop
    for segment in segList:

      # List if synapses to delete
      synsToDel = [syn for syn in segment.syns if syn[2] < minPermanence]

      if len(synsToDel) == len(segment.syns):
        segsToDel.append(segment) # will remove the whole segment
      else:
        if len(synsToDel) > 0:
          for syn in synsToDel: # remove some synapses on segment
            segment.syns.remove(syn)
            nSynsRemoved += 1
        if len(segment.syns) < minNumSyns:
          segsToDel.append(segment)

    # Remove segments that don't have enough synapses and also take them
    # out of the segment update list, if they are in there
    nSegsRemoved += len(segsToDel)
    for seg in segsToDel: # remove some segments of this cell
      self.cleanUpdatesList(colIdx, cellIdx, seg)
      self.cells[colIdx][cellIdx].remove(seg)
      nSynsRemoved += len(seg.syns)

    return nSegsRemoved, nSynsRemoved


  ################################################################################
  def trimSegments(self, minPermanence=None, minNumSyns=None):
    """ This method deletes all synapses whose permanence is less than
    minPermanence and deletes any segments that have less than
    minNumSyns synapses remaining.

    Parameters:
    --------------------------------------------------------------
    minPermanence:      Any syn whose permamence is 0 or < minPermanence will
                        be deleted. If None is passed in, then
                        self.connectedPerm is used.
    minNumSyns:         Any segment with less than minNumSyns synapses remaining
                        in it will be deleted. If None is passed in, then
                        self.activationThreshold is used.
    retval:             (numSegsRemoved, numSynsRemoved)
    """

    # Fill in defaults
    if minPermanence is None:
      minPermanence = self.connectedPerm
    if minNumSyns is None:
      minNumSyns = self.activationThreshold

    # Loop through all cells
    totalSegsRemoved, totalSynsRemoved = 0, 0
    for c,i in product(xrange(self.numberOfCols), xrange(self.cellsPerColumn)):

      (segsRemoved, synsRemoved) = self.trimSegmentsInCell(colIdx=c, cellIdx=i,
                segList=self.cells[c][i], minPermanence=minPermanence,
                minNumSyns=minNumSyns)
      totalSegsRemoved += segsRemoved
      totalSynsRemoved += synsRemoved


    return totalSegsRemoved, totalSynsRemoved

  ################################################################################
  def cleanUpdatesList(self, col, cellIdx, seg):
    """
    Removes any update that would be for the given col, cellIdx, segIdx.
    NOTE: logically, we need to do this when we delete segments, so that if
    an update refers to a segment that was just deleted, we also remove
    that update from the update list. However, I haven't seen it trigger
    in any of the unit tests yet, so it might mean that it's not needed
    and that situation doesn't occur, by construction.
    todo: check if that situation occurs.
    """
    for key, updateList in self.segmentUpdates.iteritems():
      c,i = key[0], key[1]
      if c == col and i == cellIdx:
        for update in updateList:
          if update[1].segment == seg:
            self.removeSegmentUpdate(update)

  ################################################################################
  def finishLearning(self):
    """Called when learning has been completed. This method just calls
    trimSegments. (finishLearning is here for backward compatibility)
    """

    # Keep weakly formed synapses around because they contain confidence scores
    #  for paths out of learned sequenced and produce a better prediction than
    #  chance.
    self.trimSegments(minPermanence=0.00001)


  ################################################################################
  def checkPrediction2(self, patternNZs, output=None, confidence=None,
                      details=False):

    """
    This function will replace checkPrediction.

    This function produces goodness-of-match scores for a set of input patterns, by
    checking for their presense in the current and predicted output of the TP.
    Returns a global count of the number of extra and missing bits, the
    confidence scores for each input pattern, and (if requested) the
    bits in each input pattern that were not present in the TP's prediction.

    todo: Add option to check predictedState only.

    Parameters:
    ==========
    patternNZs:       a list of input patterns that we want to check for. Each element
                      is a list of the non-zeros in that pattern.
    output:           The output of the TP. If not specified, then use the
                      TP's current output. This can be specified if you are
                      trying to check the prediction metric for an output from
                      the past.
    confidence:       The cell confidences. If not specified, then use the
                      TP's current self.confidence. This can be specified if you are
                      trying to check the prediction metrics for an output
                      from the past.
    details:          if True, also include details of missing bits per pattern.


    Return value:
    ============
    The following list is returned:

    [
      totalExtras,
      totalMissing,
      [conf_1, conf_2, ...],
      [missing1, missing2, ...]
    ]

    totalExtras:      a global count of the number of 'extras', i.e. bits that
                      are on in the current output but not in the or of all the
                      passed in patterns

    totalMissing:     a global count of all the missing bits, i.e. the bits that
                      are on in the or of the patterns, but not in the current
                      output

    conf_i            the confidence score for the i'th pattern in patternsToCheck

    missing_i         the bits in the i'th pattern that were missing
                      in the output. This list is only returned if details is
                      True.
    """


    # Get the non-zeros in each pattern
    numPatterns = len(patternNZs)

    # Compute the union of all the expected patterns
    orAll = set()
    orAll = orAll.union(*patternNZs)

    # Get the list of active columns in the output
    if output is None:
      assert self.currentOutput is not None
      output = self.currentOutput
    output = set(output.sum(axis=1).nonzero()[0])

    # Compute the total extra and missing in the output
    totalExtras = len(output.difference(orAll))
    totalMissing = len(orAll.difference(output))

    # Get the percent confidence level per column by summing the confidence levels
    #  of the cells in the column. During training, each segment's confidence
    #  number is computed as a running average of how often it correctly
    #  predicted bottom-up activity on that column. A cell's confidence number
    #  is taken from the first active segment found in the cell. Note that
    #  confidence will only be non-zero for predicted columns.
    if confidence is None:
      confidence = self.confidence['t']
    # Set the column confidence to be the max of the cell confidences in that
    #  column.
    colConfidence   = self.columnConfidences(confidence)


    # Assign confidences to each pattern
    confidences = []
    for i in xrange(numPatterns):
      # Sum of the column confidences for this pattern
      positivePredictionSum = colConfidence[patternNZs[i]].sum()
      # How many columns in this pattern
      positiveColumnCount   = len(patternNZs[i])

      # Sum of all the column confidences
      totalPredictionSum    = colConfidence.sum()
      # Total number of columns
      totalColumnCount      = len(colConfidence)

      negativePredictionSum = totalPredictionSum - positivePredictionSum
      negativeColumnCount   = totalColumnCount   - positiveColumnCount

      # Compute the average confidence score per column for this pattern
      if positiveColumnCount != 0:
        positivePredictionScore = positivePredictionSum/positiveColumnCount
      else:
        positivePredictionScore = 0.0

      # Compute the average confidence score per column for the other patterns
      if negativeColumnCount != 0:
        negativePredictionScore = negativePredictionSum/negativeColumnCount
      else:
        negativePredictionScore = 0.0
      predictionScore         = positivePredictionScore - negativePredictionScore

      confidences.append((predictionScore,
                          positivePredictionScore,
                          negativePredictionScore))

    # Include detail? (bits in each pattern that were missing from the output)
    if details:
      missingPatternBits = [set(pattern).difference(output) \
                                  for pattern in patternNZs]

      return (totalExtras, totalMissing, confidences, missingPatternBits)
    else:
      return (totalExtras, totalMissing, confidences)

  #############################################################################
  def getSegmentActivityLevel(self, seg, activeState, connectedSynapsesOnly =False):
    """This routine computes the activity level of a segment given activeState.
    It can tally up only connected synapses (permanence >= connectedPerm), or
    all the synapses of the segment, at either t or t-1.
    """

    # todo: Computing in C, use function getSegmentActivityLevel
    return getSegmentActivityLevel(seg.syns, activeState, connectedSynapsesOnly,
                                   self.connectedPerm)

  #############################################################################
  def isSegmentActive(self, seg, activeState):
    """
    A segment is active if it has >= activationThreshold connected
    synapses that are active due to activeState.

    Notes: studied various cutoffs, none of which seem to be worthwhile
           list comprehension didn't help either
    """
    # Computing in C - *much* faster
    return isSegmentActive(seg.syns, activeState,
                           self.connectedPerm, self.activationThreshold)


  ##############################################################################
  def getSegmentActiveSynapses(self, c,i,s, timeStep, newSynapses =False):

    """Return a segmentUpdate data structure containing a list of proposed changes
    to segment s. Let activeSynapses be the list of active synapses where the
    originating cells have their activeState output = 1 at time step t.
    (This list is empty if s is None since the segment doesn't exist.)
    newSynapses is an optional argument that defaults to false. If newSynapses
    is true, then newSynapseCount - len(activeSynapses) synapses are added to
    activeSynapses. These synapses are randomly chosen from the set of cells
    that have learnState = 1 at timeStep."""

    activeSynapses = []
    activeState = self.activeState[timeStep]

    if s is not None: # s can be None, if adding a new segment

      # Here we add *integers* to activeSynapses
      activeSynapses = [idx for idx, syn in enumerate(s.syns) \
                        if activeState[syn[0], syn[1]]]

    if newSynapses: # add a few more synapses

      nSynapsesToAdd = self.newSynapseCount - len(activeSynapses)

      # Here we add *pairs* (colIdx, cellIdx) to activeSynapses
      activeSynapses += self.chooseCellsToLearnFrom(c,i,s, nSynapsesToAdd, timeStep)

    # It's still possible that activeSynapses is empty, and this will
    # be handled in addToSegmentUpdates

    # NOTE: activeSynapses contains a mixture of integers and pairs of integers
    # - integers are indices of synapses already existing on the segment,
    #   that we will need to update.
    # - pairs represent source (colIdx, cellIdx) of new synapses to create on the
    #   segment
    update = TM.SegmentUpdate(c, i, s, activeSynapses)

    return update

  #############################################################################
  def getActiveSegment(self, c, i, timeStep):
    """ For a given cell, return the segment with the strongest _connected_
    activation, i.e. sum up the activations of the connected synapses of the
    segments only. That is, a segment is active only if it has enough connected
    synapses.
    """

    # todo: put back preference for sequence segments.

    nSegments = len(self.cells[c][i])

    bestActivation = self.activationThreshold
    which = -1

    for j,s in enumerate(self.cells[c][i]):

      activity = self.getSegmentActivityLevel(s, self.activeState[timeStep], connectedSynapsesOnly = True)

      if activity >= bestActivation:
        bestActivation = activity
        which = j

    if which != -1:
      return self.cells[c][i][which]
    else:
      return None


  ##############################################################################
  def chooseCellsToLearnFrom(self, c,i,s, n, timeStep):
    """Choose n random cells to learn from.

    Returns tuples of (column index, cell index).
    """

    if n <= 0:
      return []

    tmpCandidates = [] # tmp because we'll refine just below with activeSynapses

    if timeStep == 't-1':
      tmpCandidates = numpy.where(self.learnState['t-1'] == 1)
    else:
      tmpCandidates = numpy.where(self.learnState['t'] == 1)

    # Candidates can be empty at this point, in which case we return
    # an empty segment list. adaptSegments will do nothing when getting
    # that list.
    if len(tmpCandidates[0]) == 0:
      return []

    if s is None: # new segment
      cands = [syn for syn in zip(tmpCandidates[0], tmpCandidates[1])]
    else:
      # We exclude any synapse that is already in this segment.
      synapsesAlreadyInSegment = set((syn[0], syn[1]) for syn in s.syns)
      cands = [syn for syn in zip(tmpCandidates[0], tmpCandidates[1]) \
               if (syn[0], syn[1]) not in synapsesAlreadyInSegment]

    if n == 1: # so that we don't shuffle if only one is needed
      idx = self._random.getUInt32(len(cands))
      return [cands[idx]]  # col and cell idx in col

    # If we need more than one candidate
    self._random.getUInt32(10) # this required to line RNG with C++ (??)
    indices = array([j for j in range(len(cands))], dtype='uint32')
    tmp = zeros(min(n, len(indices)), dtype='uint32')
    self._random.getUInt32Sample(indices, tmp, True)
    return [cands[j] for j in tmp]

  #############################################################################
  def getBestMatchingCell(self, c, activeState):
    """Find weakly activated cell in column. Returns index and segment of most
    activated segment above minThreshold.
    """
    # Collect all cells in column c that have at least minThreshold in the most
    # activated segment
    bestActivityInCol = self.minThreshold
    bestSegIdxInCol = -1
    bestCellInCol = -1

    for i in xrange(self.cellsPerColumn):

      maxSegActivity = 0
      maxSegIdx = 0

      for j,s in enumerate(self.cells[c][i]):

        activity = self.getSegmentActivityLevel(s, activeState, connectedSynapsesOnly =False)

        if self.verbosity >= 6:
          print " Segment Activity for column ", c, " cell ", i, " segment ", " j is ", activity

        if activity > maxSegActivity:
          maxSegActivity = activity
          maxSegIdx = j

      if maxSegActivity >= bestActivityInCol:
        bestActivityInCol = maxSegActivity
        bestSegIdxInCol = maxSegIdx
        bestCellInCol = i

    if self.verbosity >= 6:
      print "Best Matching Cell In Col: ", bestCellInCol
    if bestCellInCol == -1:
      return (None, None)
    else:
      return bestCellInCol, self.cells[c][bestCellInCol][bestSegIdxInCol]

  #############################################################################
  def getBestMatchingSegment(self, c, i, activeState):
    """For the given cell, find the segment with the largest number of active
    synapses. This routine is aggressive in finding the best match. The
    permanence value of synapses is allowed to be below connectedPerm. The number
    of active synapses is allowed to be below activationThreshold, but must be
    above minThreshold. The routine returns the segment index. If no segments are
    found, then an index of -1 is returned.
    """
    maxActivity, which = self.minThreshold, -1

    for j,s in enumerate(self.cells[c][i]):
      activity = self.getSegmentActivityLevel(s, activeState,
                                              connectedSynapsesOnly=False)

      if activity >= maxActivity:
        maxActivity, which = activity, j

    if which == -1:
      return None
    else:
      return self.cells[c][i][which]

  ################################################################################
  def getLeastUsedCell(self, c):
    """For the least used cell in a column"""
    segmentsPerCell = numpy.zeros(self.cellsPerColumn, dtype='uint32')
    for i in range(self.cellsPerColumn):
      segmentsPerCell[i] = self.getNumSegmentsInCell(c,i)

    cellMinUsage = numpy.where(segmentsPerCell==segmentsPerCell.min())[0]
    # return cellMinUsage[0] # return the first cell with minimum usage

    # if multiple cells has minimum usage, randomly pick one
    self._random.getUInt32(len(cellMinUsage))
    return cellMinUsage[self._random.getUInt32(len(cellMinUsage))]

  ################################################################################
  def updateSynapses(self, segment, synapses, delta):
    """Update a set of synapses of the given segment,
      delta can be permanenceInc, or permanenceDec.

    retval:   True if synapse reached 0
    """

    reached0 = False

    if delta > 0:
      for synapse in synapses:
        segment[synapse][2] = newValue = segment[synapse][2] + delta

        # Cap synapse permanence at permanenceMax
        if newValue > self.permanenceMax:
          segment[synapse][2] = self.permanenceMax

    else:
      for synapse in synapses:
        segment[synapse][2] = newValue = segment[synapse][2] + delta

        # Cap min synapse permanence to 0 in case there is no global decay
        if newValue < 0:
          segment[synapse][2] = 0
          reached0 = True

    return reached0


  ################################################################################
  def processSegmentUpdates(self):
    """ Go through the list of accumulated segment updates and process them
    as follows:

    if the segment update is too old, remove the update
    elseif the cell that an update belongs to has it's learnState currently
          set to True (because it was predicted and just received bottom-up), then
          positively re-enforce the synapses that predicted it and remove the update.
    elseif the cell than an update belongs to just had it's predicted state
          turn off, then negatively re-enforce the synapses that predicted it
          and remove the update.
    else leave the update in the queue.

    """

    # =================================================================
    # The segmentUpdates dict has keys which are the column,cellIdx of the
    #  owner cell. The values are lists of segment updates for that cell
    removeKeys = []
    trimSegments = []
    for key, updateList in self.segmentUpdates.iteritems():

      # Get the column number and cell index of the owner cell
      c,i = key[0], key[1]

      # Should we positively or negatively re-enforce the predicting synapses?
      positiveReinforcement = None
      if self.learnState['t'][c,i] == 1:
        positiveReinforcement = True
      elif self.predictedState['t'][c,i] == 0 \
                                and self.predictedState['t-1'][c,i] == 1:
        positiveReinforcement = False

      # Process each segment for this cell. Each segment entry contains
      #  [creationDate, SegmentInfo]
      updateListKeep = []
      for (createDate, segUpdate) in updateList:

        # Decide whether to apply the update now. If update has expired, then
        # delete it. If update was created in Phase1 then apply immediately.
        # Otherwise, if update was not created during this pass then apply now.

        if self.iterationIdx - createDate > self.segUpdateValidDuration:
          # This segment has expired. Ignore this update (and hence remove it from list)
          pass
        elif (segUpdate.phase1Flag or
             self.iterationIdx > createDate) and (positiveReinforcement is not None):
          trimSegment = self.adaptSegment(segUpdate, positiveReinforcement)
          if trimSegment:
            trimSegments.append((segUpdate.columnIdx, segUpdate.cellIdx,
                                    segUpdate.segment))
        else:
          # Keep all updates that don't match the above criteria
          updateListKeep.append((createDate,segUpdate))

      self.segmentUpdates[key] = updateListKeep
      if len(updateListKeep) == 0:
        removeKeys.append(key)


    # =====================================================================
    # Clean out empty segment updates
    for key in removeKeys:
      self.segmentUpdates.pop(key)

    # =====================================================================
    # Trim segments that had synapses go to 0
    for (c, i, segment) in trimSegments:
      self.trimSegmentsInCell(c, i, [segment], minPermanence = 0.00001,
              minNumSyns = 0)



  ################################################################################
  def adaptSegment(self, segUpdate, positiveReinforcement):
    """This function applies segment update information to a segment in a
    cell.

    If positiveReinforcement is true then synapses on the active list
    get their permanence counts incremented by permanenceInc. All other synapses
    get their permanence counts decremented by permanenceDec.

    If positiveReinforcement is false, then synapses on the active list get
    their permanence counts decremented by permanenceDec. After this step,
    any synapses in segmentUpdate that do yet exist get added with a
    permanence count of initialPerm.

    Parameters:
    -----------------------------------------------------------
    segUpdate:                SegmentUpdate instance
    positiveReinforcement:    True for positive enforcement, false for
                              negative re-enforcement

    retval:                   True if some synapses were decremented to 0
                                and the segment is a candidate for trimming
    """

    # This will be set to True if detect that any syapses were decremented to
    #  0
    trimSegment = False

    # segUpdate.segment is None when creating a new segment
    c, i, segment = segUpdate.columnIdx, segUpdate.cellIdx, segUpdate.segment

    # update.activeSynapses can be empty.
    # If not, it can contain either or both integers and tuples.
    # The integers are indices of synapses to update.
    # The tuples represent new synapses to create (src col, src cell in col).
    # We pre-process to separate these various element types.
    # synToCreate is not empty only if positiveReinforcement is True.
    # NOTE: the synapse indices start at *1* to skip the segment flags.
    activeSynapses = segUpdate.activeSynapses
    synToUpdate = set([syn for syn in activeSynapses if type(syn) == int])

    if segment is not None: # modify an existing segment

      if positiveReinforcement:

        if self.verbosity >= 4:
          print "Reinforcing segment for cell[%d,%d]" %(c,i),
          segment.printSegment()

        # Update frequency and positiveActivations
        segment.positiveActivations += 1       # positiveActivations += 1
        segment.dutyCycle(active=True)

        # First, decrement synapses that are not active
        # s is a synapse *index*, with index 0 in the segment being the tuple
        # (segId, sequence segment flag). See below, creation of segments.
        lastSynIndex = len(segment.syns) - 1
        inactiveSynIndices = [s for s in xrange(0, lastSynIndex+1) \
                              if s not in synToUpdate]
        trimSegment = segment.updateSynapses(inactiveSynIndices,
                                           -self.permanenceDec)


        # Now, increment active synapses
        activeSynIndices = [syn for syn in synToUpdate if syn <= lastSynIndex]
        segment.updateSynapses(activeSynIndices, self.permanenceInc)

        # Finally, create new synapses if needed
        # syn is now a tuple (src col, src cell)
        synsToAdd = [syn for syn in activeSynapses if type(syn) != int]

        for newSyn in synsToAdd:
          segment.addSynapse(newSyn[0], newSyn[1], self.initialPerm)

        if self.verbosity >= 4:
          print "            after",
          segment.printSegment()

      else: # positiveReinforcement is False

        desc = ""
        if self.verbosity >= 4:
          print "Negatively Reinforcing %s segment for cell[%d,%d]" \
                  % (desc, c,i),
          segment.printSegment()

        # Decrease frequency count
        segment.dutyCycle(active=True)

        # We decrement all the "active" that were passed in
        trimSegment = segment.updateSynapses(synToUpdate,
                                            -self.permanenceDec)

        if self.verbosity >= 4:
          print "            after",
          segment.printSegment()

    else: # segment is None: create a new segment

      newSegment = Segment(tp=self, isSequenceSeg=segUpdate.sequenceSegment)


      # numpy.float32 important so that we can match with C++
      for synapse in activeSynapses:
        newSegment.addSynapse(synapse[0], synapse[1], self.initialPerm)

      if self.verbosity >= 3:
        print "New segment for cell[%d,%d]" %(c,i),
        newSegment.printSegment()

      self.cells[c][i].append(newSegment)


    return trimSegment

  ################################################################################
  def getSegmentInfo(self, collectActiveData = False):
    """Returns information about the distribution of segments, synapses and
    permanence values in the current TP. If requested, also returns information
    regarding the number of currently active segments and synapses.

    The method returns the following tuple:

    (
      nSegments,        # total number of segments
      nSynapses,        # total number of synapses
      nActiveSegs,      # total no. of active segments
      nActiveSynapses,  # total no. of active synapses
      distSegSizes,     # a dict where d[n] = number of segments with n synapses
      distNSegsPerCell, # a dict where d[n] = number of cells with n segments
      distPermValues,   # a dict where d[p] = number of synapses with perm = p/10
      distAges,         # a list of tuples (ageRange, numSegments)
    )

    nActiveSegs and nActiveSynapses are 0 if collectActiveData is False
    """

    nSegments, nSynapses = 0, 0
    nActiveSegs, nActiveSynapses = 0, 0
    distSegSizes, distNSegsPerCell = {}, {}
    distPermValues = {}   # Num synapses with given permanence values

    numAgeBuckets = 20
    distAges = []
    ageBucketSize = int((self.lrnIterationIdx+20) / 20)
    for i in range(numAgeBuckets):
      distAges.append(['%d-%d' % (i*ageBucketSize, (i+1)*ageBucketSize-1), 0])

    for c in xrange(self.numberOfCols):
      for i in xrange(self.cellsPerColumn):

        if len(self.cells[c][i]) > 0:
          nSegmentsThisCell = len(self.cells[c][i])
          nSegments += nSegmentsThisCell
          if distNSegsPerCell.has_key(nSegmentsThisCell):
            distNSegsPerCell[nSegmentsThisCell] += 1
          else:
            distNSegsPerCell[nSegmentsThisCell] = 1
          for seg in self.cells[c][i]:
            nSynapsesThisSeg = seg.getNumSynapses()
            nSynapses += nSynapsesThisSeg
            if distSegSizes.has_key(nSynapsesThisSeg):
              distSegSizes[nSynapsesThisSeg] += 1
            else:
              distSegSizes[nSynapsesThisSeg] = 1

            # Accumulate permanence value histogram
            for syn in seg.syns:
              p = int(syn[2]*10)
              if distPermValues.has_key(p):
                distPermValues[p] += 1
              else:
                distPermValues[p] = 1

            # Accumulate segment age histogram
            age = self.lrnIterationIdx - seg.lastActiveIteration
            ageBucket = int(age/ageBucketSize)
            distAges[ageBucket][1] += 1

            # Get active synapse statistics if requested
            if collectActiveData:
              if self.isSegmentActive(seg, self.infActiveState['t']):
                nActiveSegs += 1
              for syn in seg.syns:
                if self.activeState['t'][syn[0]][syn[1]] == 1:
                  nActiveSynapses += 1

    return (nSegments, nSynapses, nActiveSegs, nActiveSynapses,
            distSegSizes, distNSegsPerCell, distPermValues, distAges)

################################################################################
################################################################################


class Segment(object):
  """
  The Segment class is a container for all of the segment variables and
  the synapses it owns.
  """

  ## These are iteration count tiers used when computing segment duty cycle.
  dutyCycleTiers =  [0,       100,      320,    1000,
                     3200,    10000,    32000,  100000,
                     320000]

  ## This is the alpha used in each tier. dutyCycleAlphas[n] is used when
  #  `iterationIdx > dutyCycleTiers[n]`.
  dutyCycleAlphas = [None,    0.0032,    0.0010,  0.00032,
                     0.00010, 0.000032,  0.00001, 0.0000032,
                     0.0000010]


  def __init__(self, tp, isSequenceSeg):
    self.tp = tp
    self.segID = tp.segID
    tp.segID += 1

    self.isSequenceSeg = isSequenceSeg
    self.lastActiveIteration = tp.lrnIterationIdx

    self.positiveActivations = 1
    self.totalActivations = 1

    # These are internal variables used to compute the positive activations
    #  duty cycle.
    # Callers should use dutyCycle()
    self._lastPosDutyCycle = 1.0 / tp.lrnIterationIdx
    self._lastPosDutyCycleIteration = tp.lrnIterationIdx

    # Each synapse is a tuple (srcCellCol, srcCellIdx, permanence)
    self.syns = []


  def __ne__(self, s):
    return not self == s


  def __eq__(self, s):
    d1 = self.__dict__
    d2 = s.__dict__
    if set(d1) != set(d2):
      return False
    for k, v in d1.iteritems():
      if k in ('tp',):
        continue
      elif v != d2[k]:
        return False
    return True


  def dutyCycle(self, active=False, readOnly=False):
    """Compute/update and return the positive activations duty cycle of
    this segment. This is a measure of how often this segment is
    providing good predictions.

    @param active   True if segment just provided a good prediction

    @param readOnly If True, compute the updated duty cycle, but don't change
               the cached value. This is used by debugging print statements.

    @returns The duty cycle, a measure of how often this segment is
    providing good predictions.

    **NOTE:** This method relies on different schemes to compute the duty cycle
    based on how much history we have. In order to support this tiered
    approach **IT MUST BE CALLED ON EVERY SEGMENT AT EACH DUTY CYCLE TIER**
    (@ref dutyCycleTiers).

    When we don't have a lot of history yet (first tier), we simply return
    number of positive activations / total number of iterations

    After a certain number of iterations have accumulated, it converts into
    a moving average calculation, which is updated only when requested
    since it can be a bit expensive to compute on every iteration (it uses
    the pow() function).

    The duty cycle is computed as follows:

        dc[t] = (1-alpha) * dc[t-1] + alpha * value[t]

    If the value[t] has been 0 for a number of steps in a row, you can apply
    all of the updates at once using:

        dc[t] = (1-alpha)^(t-lastT) * dc[lastT]

    We use the alphas and tiers as defined in @ref dutyCycleAlphas and
    @ref dutyCycleTiers.
    """
    # For tier #0, compute it from total number of positive activations seen
    if self.tp.lrnIterationIdx <= self.dutyCycleTiers[1]:
      dutyCycle = float(self.positiveActivations) \
                                    / self.tp.lrnIterationIdx
      if not readOnly:
        self._lastPosDutyCycleIteration = self.tp.lrnIterationIdx
        self._lastPosDutyCycle = dutyCycle
      return dutyCycle

    # How old is our update?
    age = self.tp.lrnIterationIdx - self._lastPosDutyCycleIteration

    # If it's already up to date, we can returned our cached value.
    if age == 0 and not active:
      return self._lastPosDutyCycle

    # Figure out which alpha we're using
    for tierIdx in range(len(self.dutyCycleTiers)-1, 0, -1):
      if self.tp.lrnIterationIdx > self.dutyCycleTiers[tierIdx]:
        alpha = self.dutyCycleAlphas[tierIdx]
        break

    # Update duty cycle
    dutyCycle = pow(1.0-alpha, age) * self._lastPosDutyCycle
    if active:
      dutyCycle += alpha

    # Update cached values if not read-only
    if not readOnly:
      self._lastPosDutyCycleIteration = self.tp.lrnIterationIdx
      self._lastPosDutyCycle = dutyCycle

    return dutyCycle


  def printSegment(self):
    """Print segment information for verbose messaging and debugging.
    This uses the following format:

     ID:54413 True 0.64801 (24/36) 101 [9,1]0.75 [10,1]0.75 [11,1]0.75

    where:
      54413 - is the unique segment id
      True - is sequence segment
      0.64801 - moving average duty cycle
      (24/36) - (numPositiveActivations / numTotalActivations)
      101 - age, number of iterations since last activated
      [9,1]0.75 - synapse from column 9, cell #1, strength 0.75
      [10,1]0.75 - synapse from column 10, cell #1, strength 0.75
      [11,1]0.75 - synapse from column 11, cell #1, strength 0.75
    """
    # Segment ID
    print "ID:%-5d" % (self.segID),

    # Sequence segment or pooling segment
    if self.isSequenceSeg:
      print "True",
    else:
      print "False",

    # Duty cycle
    print "%9.7f" % (self.dutyCycle(readOnly=True)),

    # numPositive/totalActivations
    print "(%4d/%-4d)" % (self.positiveActivations,
                          self.totalActivations),

    # Age
    print "%4d" % (self.tp.lrnIterationIdx - self.lastActiveIteration),

    # Print each synapses on this segment as: srcCellCol/srcCellIdx/perm
    # if the permanence is above connected, put [] around the synapse info
    # For aid in comparing to the C++ implementation, print them in sorted
    #  order
    sortedSyns = sorted(self.syns)
    for _, synapse in enumerate(sortedSyns):
      print "[%d,%d]%4.2f" % (synapse[0], synapse[1], synapse[2]),
    print


  def isSequenceSegment(self):
    return self.isSequenceSeg


  def getNumSynapses(self):
    return len(self.syns)


  def freeNSynapses(self, numToFree, inactiveSynapseIndices, verbosity= 0):
    """Free up some synapses in this segment. We always free up inactive
    synapses (lowest permanence freed up first) before we start to free up
    active ones.

    @param numToFree              number of synapses to free up
    @param inactiveSynapseIndices list of the inactive synapse indices.
    """
    # Make sure numToFree isn't larger than the total number of syns we have
    assert (numToFree <= len(self.syns))

    if (verbosity >= 4):
      print "\nIn PY freeNSynapses with numToFree =", numToFree,
      print "inactiveSynapseIndices =",
      for i in inactiveSynapseIndices:
        print self.syns[i][0:2],
      print

    # Remove the lowest perm inactive synapses first
    if len(inactiveSynapseIndices) > 0:
      perms = numpy.array([self.syns[i][2] for i in inactiveSynapseIndices])
      candidates = numpy.array(inactiveSynapseIndices)[
          perms.argsort()[0:numToFree]]
      candidates = list(candidates)
    else:
      candidates = []

    # Do we need more? if so, remove the lowest perm active synapses too
    if len(candidates) < numToFree:
      activeSynIndices = [i for i in xrange(len(self.syns))
                          if i not in inactiveSynapseIndices]
      perms = numpy.array([self.syns[i][2] for i in activeSynIndices])
      moreToFree = numToFree - len(candidates)
      moreCandidates = numpy.array(activeSynIndices)[
          perms.argsort()[0:moreToFree]]
      candidates += list(moreCandidates)

    if verbosity >= 4:
      print "Deleting %d synapses from segment to make room for new ones:" % (
          len(candidates)), candidates
      print "BEFORE:",
      self.printSegment()

    # Free up all the candidates now
    synsToDelete = [self.syns[i] for i in candidates]
    for syn in synsToDelete:
      self.syns.remove(syn)

    if verbosity >= 4:
      print "AFTER:",
      self.printSegment()


  def addSynapse(self, srcCellCol, srcCellIdx, perm):
    """Add a new synapse

    @param srcCellCol source cell column
    @param srcCellIdx source cell index within the column
    @param perm       initial permanence
    """
    self.syns.append([int(srcCellCol), int(srcCellIdx), numpy.float32(perm)])


  def updateSynapses(self, synapses, delta):
    """Update a set of synapses in the segment.

    @param tp       The owner TP
    @param synapses List of synapse indices to update
    @param delta    How much to add to each permanence

    @returns   True if synapse reached 0
    """
    reached0 = False

    if delta > 0:
      for synapse in synapses:
        self.syns[synapse][2] = newValue = self.syns[synapse][2] + delta

        # Cap synapse permanence at permanenceMax
        if newValue > self.tp.permanenceMax:
          self.syns[synapse][2] = self.tp.permanenceMax

    else:
      for synapse in synapses:
        self.syns[synapse][2] = newValue = self.syns[synapse][2] + delta

        # Cap min synapse permanence to 0 in case there is no global decay
        if newValue <= 0:
          self.syns[synapse][2] = 0
          reached0 = True

    return reached0
