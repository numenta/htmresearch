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

import random
import numpy
import copy
import json

from nupic.engine import Network
from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.frameworks.layers.laminar_network import createNetwork


class FeedbackExperiment(object):
  """
  Class for running simple feedback experiments.

  These experiments use a laminar network to test out various properties of
  inference and learning using a sensors and a network with feedback.
  """

  def __init__(self,
               numCorticalColumns=1,
               inputSize=2048,
               numInputBits=40,
               L2Overrides=None,
               L4Overrides=None,
               numLearningPasses=4,
               seed=42):
    """
    Creates the network and initialize the experiment.

    Parameters:
    ----------------------------
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
    """
    registerAllResearchRegions()

    self.numLearningPoints = numLearningPasses
    self.numColumns = numCorticalColumns
    self.inputSize = inputSize
    self.numInputBits = numInputBits



    # self.L4RegionType = "py.ExtendedTMRegion"
    self.L4RegionType = "py.ApicalTMRegion"



    # seed
    self.seed = seed
    random.seed(seed)
    # update parameters with overrides
    self.config = {
      #"networkType": "MultipleL4L2Columns",
      "numCorticalColumns": numCorticalColumns,
      "externalInputSize": 0,
      "sensorInputSize": inputSize,
      #   NOTE: To switch between ExtendedTMRegion and ApicalTMRegion:
      # 1- change l4regiontype below
      # 2- change the configuration set below
      # 3- change the linking between current active cells and basal input,
      # in l2_l4_network_creation.py,
      # which only should be there for ApicalTMRegion
      "L4RegionType": self.L4RegionType,#"py.ExtendedTMRegion",
      #"L4RegionType": "py.ExtendedTMRegion",
      "L4Params": self.getDefaultL4Params(inputSize),
      "L2Params": self.getDefaultL2Params(inputSize),
    }

    if L2Overrides is not None:
      self.config["L2Params"].update(L2Overrides)

    if L4Overrides is not None:
      self.config["L4Params"].update(L4Overrides)

    # create network
    self.network = self.myCreateNetwork(self.config)

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





  def myCreateNetwork(self, networkConfig):

        suffix = '_0'
        network = Network()

        sensorInputName = "sensorInput" + suffix
        L4ColumnName = "L4Column" + suffix
        L2ColumnName = "L2Column" + suffix

        L4Params = copy.deepcopy(networkConfig["L4Params"])

        # The different assumptions for ApicalTMRegion and ExtendedTMRegion....
        if networkConfig["L4RegionType"] == "py.ApicalTMRegion":
            L4Params["basalInputWidth"] = networkConfig["L4Params"]["columnCount"] * networkConfig["L4Params"]["cellsPerColumn"]
        elif networkConfig["L4RegionType"] == "py.ExtendedTMRegion":
            L4Params["basalInputWidth"] = networkConfig["externalInputSize"]
        else:
            raise Exception("Invalid L4 Region Type!")

        L4Params["apicalInputWidth"] = networkConfig["L2Params"]["cellCount"]

        network.addRegion(
          sensorInputName, "py.RawSensor",
          json.dumps({"outputWidth": networkConfig["sensorInputSize"]}))

        network.addRegion(
          L4ColumnName, networkConfig["L4RegionType"],
          json.dumps(L4Params))
        network.addRegion(
          L2ColumnName, "py.ColumnPoolerRegion",
          json.dumps(networkConfig["L2Params"]))

        network.setPhases(sensorInputName,[0])


        # L4 and L2 regions always have phases 2 and 3, respectively
        network.setPhases(L4ColumnName,[2])
        network.setPhases(L2ColumnName,[3])

        network.link(sensorInputName, L4ColumnName, "UniformLink", "",
                         srcOutput="dataOut", destInput="activeColumns")

        # Link L4 to L2
        network.link(L4ColumnName, L2ColumnName, "UniformLink", "",
                     srcOutput="activeCells", destInput="feedforwardInput")
        network.link(L4ColumnName, L2ColumnName, "UniformLink", "",
                     srcOutput="predictedActiveCells",
                     destInput="feedforwardGrowthCandidates")

        # Link L2 feedback to L4
        network.link(L2ColumnName, L4ColumnName, "UniformLink", "",
                     srcOutput="feedForwardOutput", destInput="apicalInput",
                     propagationDelay=1)

        # # ONLY for ApicalTM: link the region to itself laterally (basally)
        if networkConfig["L4RegionType"] == "py.ApicalTMRegion":
            network.link(L4ColumnName, L4ColumnName, "UniformLink", "",
                     srcOutput="activeCells", destInput="basalInput",
                     propagationDelay=1)
            network.link(L4ColumnName, L4ColumnName, "UniformLink", "", srcOutput="winnerCells", destInput="basalGrowthCandidates",propagationDelay=1)


        # Link reset output to L2. For L4, an empty input is sufficient for a reset.
        network.link(sensorInputName, L2ColumnName, "UniformLink", "",
                     srcOutput="resetOut", destInput="resetIn")

        #enableProfiling(network)
        for region in network.regions.values():
            region.enableProfiling()

        return network




  def learnSequences(self, sequences):
    """
    Learns all provided sequences. Always reset the network in between
    sequences.

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
    """
    # This method goes through four phases:
    #   1) We first train L4 on the sequences, over multiple passes
    #   2) We then train L2 in one pass.
    #   3) We then continue training on L4 so the apical segments learn
    #   4) We run inference to store L2 representations for each sequence
    # retrieve L2 representations

    # print "1) Train L4 sequence memory"


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

        # Reset signal
        self.sensorInputs[0].addDataToQueue([], 1, 0)
        iterations += 1

      if iterations > 0:
        self.network.run(iterations)



    # This generates some bugs wrt the first sequence seen during learning...
    # self._disableL2()
    # self._setLearningMode(l4Learning=True, l2Learning=False)
    # # Run multiple passes through each sequence
    # for zzz in xrange(self.numLearningPoints):
    #   myseqs = list(sequences)
    #   if zzz == 0:
    #     myseqs = list(sequences)[1:]
    #     random.shuffle(myseqs)
    #     myseqs = [sequences[0]] + myseqs
    #   else:
    #     random.shuffle(myseqs)
    #   for sequenceNum, sequence in enumerate(myseqs): #sequences):
    #
    #     # keep track of numbers of iterations to run for this sequence
    #     iterations = 0
    #
    #
    #     for s in sequence:
    #       self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
    #       iterations += 1
    #
    #     # Reset signal
    #     self.sensorInputs[0].addDataToQueue([], 1, 0)
    #     iterations += 1
    #
    #     if iterations > 0:
    #       self.network.run(iterations)
    #
    #     self.sendReset() # Reset after seeing each sequence, just in case


    # print "2) Train L2"
    self._enableL2()
    self._setLearningMode(l4Learning=False, l2Learning=True)
    for sequenceNum, sequence in enumerate(sequences):
      for s in sequence:
        self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
        self.network.run(1)
      self.sendReset()

    # print "3) Train L4 apical segments"
    self._setLearningMode(l4Learning=True, l2Learning=False)
    for p in range(5):
      for sequenceNum, sequence in enumerate(sequences):
        for s in sequence:
          self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
          self.network.run(1)
        self.sendReset()

    # Re-run the sequences once each and store L2 representations for each
    self._setLearningMode(l4Learning=False, l2Learning=False)
    for sequenceNum, sequence in enumerate(sequences):
      for s in sequence:
        self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
        self.network.run(1)
      self.objectL2Representations[sequenceNum] = self.getL2Representations()
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
    if enableFeedback is False:
      self._disableL2()
    else:
      self._enableL2()

    self._setLearningMode(l4Learning=False, l2Learning=False)

    if sequenceNumber is not None:
      if sequenceNumber not in self.objectL2Representations:
        raise ValueError("The provided sequence was not given during"
                         " learning")

    L2Responses=[]
    L4Responses=[]
    L4Predictive=[]
    activityTrace = numpy.zeros(len(sequence))

    totalActiveCells = 0
    totalPredictedActiveCells = 0
    for i,s in enumerate(sequence):
      self.sensorInputs[0].addDataToQueue(list(s), 0, 0)
      self.network.run(1)

      activityTrace[i] = len(self.getL4Representations()[0])
      L4Responses.append(self.getL4Representations()[0])
      L4Predictive.append(self.getL4PredictiveCells()[0])
      L2Responses.append(self.getL2Representations()[0])
      if i >= burnIn:
        totalActiveCells += len(self.getL4Representations()[0])
        totalPredictedActiveCells += len(self.getL4PredictedActiveCells()[0])

    if reset:
      self.sendReset()

    avgActiveCells = float(totalActiveCells) / len(sequence)
    avgPredictedActiveCells = float(totalPredictedActiveCells) / len(sequence)

    responses = {
            "L2Responses": L2Responses,
            "L4Responses": L4Responses,
            "L4Predictive": L4Predictive
            }
    return avgActiveCells,avgPredictedActiveCells,activityTrace, responses


  def sendReset(self, sequenceId=0):
    """
    Sends a reset signal to the network.
    """
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
    # ApicalTMRegion uses "getPredictedCells", while ExtendedTMRegion uses "getPredictiveCells".
    #return [set(column._tm.getPredictiveCells()) for column in self.L4Columns]
    if self.L4RegionType == "py.ApicalTMRegion":
        return [set(column._tm.getPredictedCells()) for column in self.L4Columns]
    elif self.L4RegionType == "py.ExtendedTMRegion":
        return [set(column._tm.getPredictiveCells()) for column in self.L4Columns]
    else:
        raise (Exception("Invalid L4 Region Type!"))


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

    if self.L4RegionType == "py.ApicalTMRegion":
        return {
            "columnCount": inputSize,
            "cellsPerColumn": 8,
            "learn": True,
            "initialPermanence": 0.51,
            "connectedPermanence": 0.6,
            "permanenceIncrement": 0.1,
            "permanenceDecrement": 0.02,
            "minThreshold": 13, #13,
            "basalPredictedSegmentDecrement": 0.0,
            "apicalPredictedSegmentDecrement": 0.0,
            "activationThreshold": 15, #15,
            "sampleSize": 20, #60,  # 1.5 * 40
            "implementation": "ApicalModulation", #"ApicalTiebreak",
            # "implementation": "ApicalTiebreak",
            "seed": self.seed
            }
    elif self.L4RegionType == "py.ExtendedTMRegion":
        return{
              "columnCount": inputSize,
              "cellsPerColumn": 8,
              "formInternalBasalConnections": True,
              "learn": True,
              "learnOnOneCell": False,
              "initialPermanence": 0.51,
              "connectedPermanence": 0.6,
              "permanenceIncrement": 0.1,
              "permanenceDecrement": 0.02,
              "minThreshold": 13,
              "predictedSegmentDecrement": 0.00,
              "activationThreshold": 15,
              "maxNewSynapseCount": 20,
              "implementation": "etm",
              "seed": self.seed
              }
    else:
        raise(Exception("Invalid L4 Region Type! (current value: "+self.L4RegionType+")"))


  def getDefaultL2Params(self, inputSize):
    """
    Returns a good default set of parameters to use in the L4 region.
    """
    return {
      "cellCount": 2048,
      "inputWidth": inputSize * 8,
      "learningMode": True,
      "sdrSize": 40,
      "synPermProximalInc": 0.1,
      "synPermProximalDec": 0.001,
      "initialProximalPermanence": 0.6,
      "minThresholdProximal": 10,
      "sampleSizeProximal": 20,
      "connectedPermanenceProximal": 0.5,
      "synPermDistalInc": 0.1,
      "synPermDistalDec": 0.02,
      "initialDistalPermanence": 0.41,
      "activationThresholdDistal": 13,
      "sampleSizeDistal": 20,
      "connectedPermanenceDistal": 0.5,
      "distalSegmentInhibitionFactor": 1.5,
      "seed": self.seed,
    }


  def _setLearningMode(self, l4Learning = False, l2Learning=False):
    """
    Sets the learning mode for L4 and L2.
    """
    for column in self.L4Columns:
      column.setParameter("learn", 0, l4Learning)
    for column in self.L2Columns:
      column.setParameter("learningMode", 0, l2Learning)


  def _disableL2(self):
    self.network.setMaxEnabledPhase(2)


  def _enableL2(self):
    self.network.setMaxEnabledPhase(3)
