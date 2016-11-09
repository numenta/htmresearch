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

"""
This class supports using an L4-L2 network for experiments with feedback
and pure temporal sequences.
"""

import os
import random
import inspect
import cPickle

from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.frameworks.layers.laminar_network import createNetwork


def rerunExperimentFromLogfile(logFilename):
  """
  Create an experiment class according to the sequence of operations in logFile
  and return resulting experiment instance.
  """
  with open(logFilename,"rb") as f:
    callLog = cPickle.load(f)

  # Assume first one is call to constructor
  exp = FeedbackExperiment(**callLog[0][1])

  # Call subsequent methods, using stored parameters
  for call in callLog[1:]:
    method = getattr(exp, call[0])
    method(**call[1])

  return exp


class FeedbackExperiment(object):
  """
  Feedback experiment.

  This experiment uses a laminar network to test out various properties of
  inference and learning using a sensors and a network with feedback.

  """

  def __init__(self,
               name,
               numCorticalColumns=1,
               inputSize=2048,
               numInputBits=40,
               L2Overrides=None,
               L4Overrides=None,
               numLearningPasses=4,
               seed=42,
               logCalls = False):
    """
    Creates the network.

    Parameters:
    ----------------------------
    @param   name (str)
             Experiment name

    @param   numCorticalColumns (int)
             Number of cortical columns in the network

    @param   inputSize (int)
             Size of the sensory input

    @param   numInputBits (int)
             Number of ON bits in the generated input patterns

    @param   L2Overrides (dict)
             Parameters to override in the L2 region

    @param   L4Overrides
             Parameters to override in the L4 region

    @param   numLearningPasses (int)
             Number of times each pair should be seen to be learnt

    @param   logCalls (bool)
             If true, calls to main functions will be logged internally. The
             log can then be saved with saveLogs(). This allows us to recreate
             the complete network behavior using rerunExperimentFromLogfile
             which is very useful for debugging.

    """
    # Handle logging - this has to be done first
    self.callLog = []
    self.logCalls = logCalls
    if self.logCalls:
      frame = inspect.currentframe()
      args, _, _, values = inspect.getargvalues(frame)
      values.pop('frame')
      values.pop('self')
      self.callLog.append([inspect.getframeinfo(frame)[2], values])

    registerAllResearchRegions()
    self.name = name

    self.numLearningPoints = numLearningPasses
    self.numColumns = numCorticalColumns
    self.inputSize = inputSize
    self.numInputBits = numInputBits

    # seed
    self.seed = seed
    random.seed(seed)

    # update parameters with overrides
    self.config = {
      "networkType": "MultipleL4L2Columns",
      "numCorticalColumns": numCorticalColumns,
      "externalInputSize": 0,
      "sensorInputSize": inputSize,
      "L4Params": self.getDefaultL4Params(inputSize),
      "L2Params": self.getDefaultL2Params(inputSize),
    }

    if L2Overrides is not None:
      self.config["L2Params"].update(L2Overrides)

    if L4Overrides is not None:
      self.config["L4Params"].update(L4Overrides)

    # create network
    self.network = createNetwork(self.config)

    # We have to explicitly initialize if we are going to change the phases
    self.network.initialize()

    self.sensorInputs = []
    self.L4Columns = []
    self.L2Columns = []

    for i in xrange(self.numColumns):
      self.sensorInputs.append(
        self.network.regions["sensorInput_" + str(i)].getSelf()
      )
      self.L4Columns.append(
        self.network.regions["L4Column_" + str(i)].getSelf()
      )
      self.L2Columns.append(
        self.network.regions["L2Column_" + str(i)].getSelf()
      )

    # will be populated during training
    self.objectL2Representations = {}
    self.statistics = []


  def learnSequences(self, sequences, reset=True):
    """
    Learns all provided sequences, and optionally resets the network in between.

    Sequences format:

    sequences = [
      [
        set([16, 22, 32]),  # S0, position 0
        set([13, 15, 33])   # S0, position 1
      ],
      [
        set([6, 12, 52]),  # S1, position 0
        set([6, 2, 15])    # S1, position 1
      ],
    ]

    Note that the order of each sequence is important. It denotes the sequence
    number and will be used during inference to plot accuracy.

    Parameters:
    ----------------------------
    @param   sequences (list)
             Sequences to learn, in the canonical format specified above

    @param   reset (bool)
             If set to True (which is the default value), the network will
             be reset after learning.

    """
    # Handle logging - this has to be done first
    if self.logCalls:
      frame = inspect.currentframe()
      args, _, _, values = inspect.getargvalues(frame)
      values.pop('frame')
      values.pop('self')
      self.callLog.append([inspect.getframeinfo(frame)[2], values])

    # This method goes through four phases:
    #   1) We first train L4 on the sequences, over multiple passes
    #   2) We then train L2 in one pass.
    #   3) We then continue training on L4 so the apical segments learn
    #   4) We run inference to store L2 representations for each sequence
    # retrieve L2 representations

    print "1) Train L4 sequence memory"
    self._disableL2()
    self._setLearningMode(l4Learning=True, l2Learning=False)
    for sequenceNum, sequence in enumerate(sequences):

      # keep track of numbers of iterations to run for this sequence
      iterations = 0

      # Run multiple passes through each sequence
      for _ in xrange(self.numLearningPoints):

        for s in sequence:
          self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
          iterations += 1

        self.sensorInputs[0].addDataToQueue([], 1, 0)
        iterations += 1

      if iterations > 0:
        self.network.run(iterations)

    print "2) Train L2"
    self._enableL2()
    self._setLearningMode(l4Learning=False, l2Learning=True)
    for sequenceNum, sequence in enumerate(sequences):
      for s in sequence:
        self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
        self.network.run(1)
      self.sendReset()

    print "3) Train L4 apical segments"
    self._setLearningMode(l4Learning=True, l2Learning=False)
    for p in range(5):
      for sequenceNum, sequence in enumerate(sequences):
        for s in sequence:
          self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
          self.network.run(1)
        self.sendReset()

    # Re-run the sequences once each and store L2 representations for each
    print "Retrieving L2 representations"
    self._setLearningMode(l4Learning=False, l2Learning=False)
    for sequenceNum, sequence in enumerate(sequences):
      for s in sequence:
        self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
        self.network.run(1)

      self.objectL2Representations[sequenceNum] = self.getL2Representations()
      print sequenceNum, "L2 representation size=",len(self.objectL2Representations[sequenceNum][0])
      # print sequenceNum, self.objectL2Representations[sequenceNum]

      self.sendReset()



  def infer(self, sequence, reset=True, sequenceNumber=None, burnIn=2,
            enableFeedback=True):
    """
    Infer on a single given sequence. Sequence format:

    sequence = [
      set([16, 22, 32]),  # Position 0
      set([13, 15, 33])   # Position 1
    ]

    Parameters:
    ----------------------------
    @param   sequence (list)
             Sequence to infer, in the canonical format specified above

    @param   reset (bool)
             If set to True (which is the default value), the network will
             be reset after inference.

    @param   sequenceNumber (int)
             Number of the sequence (must match the number given during
             learning).

    @param   burnIn (int)
             Number of patterns to wait within a sequence before computing
             accuracy figures

    """
    # Handle logging - this has to be done first
    if self.logCalls:
      frame = inspect.currentframe()
      args, _, _, values = inspect.getargvalues(frame)
      values.pop('frame')
      values.pop('self')
      self.callLog.append([inspect.getframeinfo(frame)[2], values])

    if enableFeedback is False:
      self._disableL2()
    else:
      self._enableL2()

    self._setLearningMode(l4Learning=False, l2Learning=False)

    if sequenceNumber is not None:
      if sequenceNumber not in self.objectL2Representations:
        raise ValueError("The provided sequence was not given during"
                         " learning")

    totalActiveCells = 0
    totalPredictedActiveCells = 0
    for i,s in enumerate(sequence):
      self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
      self.network.run(1)

      if i >= burnIn:
        totalActiveCells += len(self.getL4Representations()[0])
        totalPredictedActiveCells += len(self.getL4PredictedActiveCells()[0])

    if reset:
      self.sendReset()

    avgActiveCells = float(totalActiveCells) / len(sequence)
    avgPredictedActiveCells = float(totalPredictedActiveCells) / len(sequence)
    # print "Sequence length=",len(sequence)
    # print "totalActiveCells=",totalActiveCells,"totalPredictedActiveCells=",totalPredictedActiveCells
    # print "avgActiveCells=",avgActiveCells,"avgPredictedActiveCells",avgPredictedActiveCells

    return totalActiveCells,totalPredictedActiveCells,avgActiveCells,avgPredictedActiveCells


  def sendReset(self, sequenceId=0):
    """
    Sends a reset signal to the network.
    """
    # Handle logging - this has to be done first
    if self.logCalls:
      frame = inspect.currentframe()
      args, _, _, values = inspect.getargvalues(frame)
      values.pop('frame')
      values.pop('self')
      (_, filename,
       _, _, _, _) = inspect.getouterframes(inspect.currentframe())[1]
      if os.path.splitext(os.path.basename(__file__))[0] != \
         os.path.splitext(os.path.basename(filename))[0]:
        self.callLog.append([inspect.getframeinfo(frame)[2], values])

    for col in xrange(self.numColumns):
      self.sensorInputs[col].addResetToQueue(sequenceId)
    self.network.run(1)


  def getL4Representations(self):
    """
    Returns the active representation in L4.
    """
    return [set(column._tm.getActiveCells()) for column in self.L4Columns]


  def getL4PredictiveCells(self):
    """
    Returns the predictive cells in L4.
    """
    return [set(column._tm.getPredictiveCells()) for column in self.L4Columns]


  def getL4PredictedActiveCells(self):
    """
    Returns the predicted active cells in each column in L4.
    """
    predictedActive = []
    for i in xrange(self.numColumns):
      region = self.network.regions["L4Column_" + str(i)]
      predictedActive.append(
        region.getOutputData("predictedActiveCells").nonzero()[0])
    return predictedActive


  def getL2Representations(self):
    """
    Returns the active representation in L2.
    """
    return [set(column._pooler.getActiveCells()) for column in self.L2Columns]


  def getDefaultL4Params(self, inputSize):
    """
    Returns a good default set of parameters to use in the L4 region.
    """
    return {
      "columnCount": inputSize,
      "cellsPerColumn": 16,
      "formInternalBasalConnections": True,
      "learningMode": True,
      "inferenceMode": True,
      "learnOnOneCell": False,
      "initialPermanence": 0.51,
      "connectedPermanence": 0.6,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "minThreshold": 13,
      "predictedSegmentDecrement": 0.01,
      "activationThreshold": 15,
      "maxNewSynapseCount": 20,
      "defaultOutputType": "predictedActiveCells",
      "implementation": "etm_cpp",
      "seed": self.seed
    }


  def getDefaultL2Params(self, inputSize):
    """
    Returns a good default set of parameters to use in the L4 region.
    """
    return {
      "columnCount": 2048,
      "inputWidth": inputSize * 16,
      "learningMode": True,
      "inferenceMode": True,
      "initialPermanence": 0.41,
      "connectedPermanence": 0.5,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "numActiveColumnsPerInhArea": 40,
      "synPermProximalInc": 0.1,
      "synPermProximalDec": 0.001,
      "initialProximalPermanence": 0.6,
      "minThresholdDistal": 10,
      "minThresholdProximal": 10,
      "predictedSegmentDecrement": 0.002,
      "activationThresholdDistal": 13,
      "maxNewProximalSynapseCount": 20,
      "maxNewDistalSynapseCount": 20,
      "maxSynapsesPerDistalSegment": 255,
      "maxSynapsesPerProximalSegment": 2000,
      "seed": self.seed
    }


  def saveLog(self, logFilename):
    """
    Save the call log history into this file.

    @param  logFilename (path)
            Filename in which to save a pickled version of the call logs.

    """
    with open(logFilename,"wb") as f:
      cPickle.dump(self.callLog,f)


  def _setLearningMode(self, l4Learning = False, l2Learning=False):
    """
    Sets the learning mode for L4 and L2.
    """
    for column in self.L4Columns:
      column.setParameter("learningMode", 0, l4Learning)
    for column in self.L2Columns:
      column.setParameter("learningMode", 0, l2Learning)


  def _disableL2(self):
    self.network.setMaxEnabledPhase(1)


  def _enableL2(self):
    self.network.setMaxEnabledPhase(2)

