#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

"""Implementations of various synapse learning rules"""


from abc import ABCMeta, abstractmethod

import numpy as np



class SynapseLearningRules(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def learnOnExistingSegments(self, connections, learningSegments, activeAxons,
                              growthCandidateAxons, **kwargs):
    """
    For each specified segment, reinforce / punish synapses according to whether
    they're active, and maybe grow new synapses.

    @param connections (any connections class)
    Connections instance that will be modified

    @param learningSegments (numpy array)
    The segments to learn on

    @param activeAxons (numpy array)
    The active input to this Connections

    @param growthCandidateAxons (numpy array)
    The (presumably active) input bits that are available for growth

    @param kwargs
    Additional arguments for specific SynapseLearningRules implementations
    """


  @abstractmethod
  def learnOnNewSegments(self, connections, learningCells, growthCandidateAxons,
                         **kwargs):
    """
    Grow segments on the specified cells and grow synapses on each segment.

    @param connections (any connections class)
    Connections instance that will be modified

    @param learningCells (numpy array)
    The cells that should grow segments

    @param growthCandidateAxons (numpy array)
    The (presumably active) input bits that are available for growth

    @param kwargs
    Additional arguments for specific SynapseLearningRules implementations
    """


class ConnectToActiveAxons(SynapseLearningRules):
  """
  Grow synapses to every active axon.
  """

  def __init__(self, initialPermanence, permanenceIncrement,
               permanenceDecrement):

    self.initialPermanence = initialPermanence
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement


  def learnOnExistingSegments(self, connections, learningSegments, activeAxons,
                              growthCandidateAxons, **kwargs):

    connections.adjustSynapses(learningSegments, activeAxons,
                               self.permanenceIncrement,
                               -self.permanenceDecrement)

    connections.growSynapses(learningSegments, growthCandidateAxons,
                             self.initialPermanence)


  def learnOnNewSegments(self, connections, learningCells, growthCandidateAxons,
                         **kwargs):

    newSegments = connections.createSegments(learningCells)
    connections.growSynapses(newSegments, growthCandidateAxons,
                             self.initialPermanence)



class ConnectToActiveAxonsWithMaxSynapseCount(SynapseLearningRules):
  """
  Grow synapses to every active axon, but stay below a "maxSynapsesPerSegment".

  When a segment has "maxSynapsesPerSegment" synapses, don't grow any new
  synapses on the segment.
  """

  def __init__(self, initialPermanence, permanenceIncrement,
               permanenceDecrement, maxSynapsesPerSegment):
    self.initialPermanence = initialPermanence
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.maxSynapsesPerSegment = maxSynapsesPerSegment


  def learnOnExistingSegments(self, connections, learningSegments, activeAxons,
                              growthCandidateAxons, rng, **kwargs):
    """

    @param kwargs
    These aren't used. Other learning rules take additional parameters like
    "potentialOverlaps".

    """

    connections.adjustSynapses(learningSegments, activeAxons,
                               self.permanenceIncrement,
                               -self.permanenceDecrement)

    maxNew = len(growthCandidateAxons)
    synapseCounts = connections.mapSegmentsToSynapseCounts(learningSegments)
    numSynapsesToReachMax = self.maxSynapsesPerSegment - synapseCounts
    maxNewBySegment = np.where(maxNew <= numSynapsesToReachMax,
                               maxNew, numSynapsesToReachMax)

    connections.growSynapsesToSample(learningSegments, growthCandidateAxons,
                                     maxNewBySegment, self.initialPermanence,
                                     rng)


  def learnOnNewSegments(self, connections, learningCells, growthCandidateAxons,
                         rng, **kwargs):

    numNewSynapses = min(len(growthCandidateAxons), self.maxSynapsesPerSegment)

    newSegments = connections.createSegments(learningCells)
    connections.growSynapsesToSample(newSegments, growthCandidateAxons,
                                     numNewSynapses, self.initialPermanence,
                                     rng)



class ConnectToSampleOfActiveAxons(SynapseLearningRules):
  """
  Aim for "sampleSize" active synapses per segment. If a segment has fewer than
  "sampleSize" active synapses, grow synapses to make up the difference.

  Stay below a "maxSynapsesPerSegment". When a segment has
  "maxSynapsesPerSegment" synapses, don't grow any new synapses on the segment.
  """

  def __init__(self, sampleSize, initialPermanence, permanenceIncrement,
               permanenceDecrement, maxSynapsesPerSegment):
    self.sampleSize = sampleSize
    self.initialPermanence = initialPermanence
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.maxSynapsesPerSegment = maxSynapsesPerSegment


  def learnOnExistingSegments(self, connections, learningSegments, activeAxons,
                              growthCandidateAxons, rng, potentialOverlaps,
                              **kwargs):

    connections.adjustSynapses(learningSegments, activeAxons,
                               self.permanenceIncrement,
                               -self.permanenceDecrement)

    maxNewBySegment = self.sampleSize - potentialOverlaps[learningSegments]

    if self.maxSynapsesPerSegment != -1:
      synapseCounts = connections.mapSegmentsToSynapseCounts(learningSegments)
      numSynapsesToReachMax = self.maxSynapsesPerSegment - synapseCounts
      maxNewBySegment = np.where(maxNewBySegment <= numSynapsesToReachMax,
                                 maxNewBySegment, numSynapsesToReachMax)

    connections.growSynapsesToSample(learningSegments, growthCandidateAxons,
                                     maxNewBySegment, self.initialPermanence,
                                     rng)


  def learnOnNewSegments(self, connections, learningCells,
                         growthCandidateAxons, rng, **kwargs):

    numNewSynapses = min(len(growthCandidateAxons), self.sampleSize)

    if self.maxSynapsesPerSegment != -1:
      numNewSynapses = min(numNewSynapses, self.maxSynapsesPerSegment)

    newSegments = connections.createSegments(learningCells)
    connections.growSynapsesToSample(newSegments, growthCandidateAxons,
                                     numNewSynapses, self.initialPermanence,
                                     rng)


def createSynapseLearningRules(**kwargs):
  """
  Factory for SynapseLearningRules
  """
  sampleSize = kwargs.get("sampleSize", -1)
  maxSynapsesPerSegment = kwargs.get("maxSynapsesPerSegment", -1)
  initialPermanence = kwargs["initialPermanence"]
  permanenceIncrement = kwargs["permanenceIncrement"]
  permanenceDecrement = kwargs["permanenceDecrement"]

  if sampleSize == -1:
    if maxSynapsesPerSegment == -1:
      return ConnectToActiveAxons(initialPermanence, permanenceIncrement,
                                  permanenceDecrement)
    else:
      return ConnectToActiveAxonsWithMaxSynapseCount(initialPermanence,
                                                     permanenceIncrement,
                                                     permanenceDecrement,
                                                     maxSynapsesPerSegment)
  else:
    return ConnectToSampleOfActiveAxons(sampleSize, initialPermanence,
                                        permanenceIncrement,
                                        permanenceDecrement,
                                        maxSynapsesPerSegment)
