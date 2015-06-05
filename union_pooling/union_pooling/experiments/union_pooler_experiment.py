# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import time

import numpy

from nupic.algorithms.KNNClassifier import KNNClassifier
from nupic.bindings.math import GetNTAReal
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
    TemporalMemoryMonitorMixin)

from sensorimotor.fast_general_temporal_memory import (
     FastGeneralTemporalMemory)
from sensorimotor.temporal_pooler_monitor_mixin import (
     TemporalPoolerMonitorMixin)
from union_pooling.union_pooler import UnionPooler



class MonitoredFastGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                         FastGeneralTemporalMemory):
  pass



# Implement a UnionPoolerMonitorMixin if needed...
class MonitoredUnionPooler(TemporalPoolerMonitorMixin, UnionPooler):
  pass



realDType = GetNTAReal()



class UnionPoolerExperiment(object):
  """
  This class defines a Temporal Memory-Union Pooler network and provides methods
  to run the network on data sequences.
  """


  DEFAULT_TEMPORAL_MEMORY_PARAMS = {"columnDimensions": (1024,),
                                    "cellsPerColumn": 8,
                                    "activationThreshold": 20,
                                    "initialPermanence": 0.5,
                                    "connectedPermanence": 0.6,
                                    "minThreshold": 20,
                                    "maxNewSynapseCount": 30,
                                    "permanenceIncrement": 0.10,
                                    "permanenceDecrement": 0.02,
                                    "seed": 42,
                                    "learnOnOneCell": False}


  DEFAULT_UNION_POOLER_PARAMS = {# Spatial Pooler Params
                                 # inputDimensions set to TM cell count
                                 # potentialRadius set to TM cell count
                                 "columnDimensions": [1024],
                                 "numActiveColumnsPerInhArea": 20,
                                 "stimulusThreshold": 0,
                                 "synPermInactiveDec": 0.01,
                                 "synPermActiveInc": 0.1,
                                 "synPermConnected": 0.1,
                                 "potentialPct": 0.5,
                                 "globalInhibition": True,
                                 "localAreaDensity": -1,
                                 "minPctOverlapDutyCycle": 0.001,
                                 "minPctActiveDutyCycle": 0.001,
                                 "dutyCyclePeriod": 1000,
                                 "maxBoost": 10.0,
                                 "seed": 42,
                                 "spVerbosity": 0,
                                 "wrapAround": True,

                                 # Union Pooler Params
                                 "activeOverlapWeight": 1.0,
                                 "predictedActiveOverlapWeight": 0.0,
                                 "fixedPoolingActivationBurst": False,
                                 "exciteFunction": None,
                                 "decayFunction": None,
                                 "maxUnionActivity": 0.20}

  DEFAULT_CLASSIFIER_PARAMS = {
    # TODO: Add parameters
  'distThreshold': 0.000001,
  'maxCategoryCount': 10,
  #'distanceMethod': 'rawOverlap',  # Default is Euclidean distance
   }


  def __init__(self, tmOverrides=None, upOverrides=None, seed=42):
    print "Initializing Temporal Memory..."
    params = dict(self.DEFAULT_TEMPORAL_MEMORY_PARAMS)
    params.update(tmOverrides or {})
    params["seed"] = seed
    self.tm = MonitoredFastGeneralTemporalMemory(mmName="TM", **params)

    print "Initializing Union Pooler..."
    params = dict(self.DEFAULT_UNION_POOLER_PARAMS)
    params.update(upOverrides or {})
    params["inputDimensions"] = [self.tm.numberOfCells()]
    params["potentialRadius"] = self.tm.numberOfCells()
    params["seed"] = seed
    self.up = MonitoredUnionPooler(mmName="UP", **params)

    print "Initializing KNN Classifier..."
    params = dict(self.DEFAULT_CLASSIFIER_PARAMS)
    self.classifier = KNNClassifier(**params)


  def runNetworkOnSequence(self, sensorSequences, sequencesLabels, tmLearn=True,
                           upLearn=None, verbosity=0, progressInterval=None):
    """
    Runs Union Pooler network on specified sequence.

    @param sensorSequences        A sequence of sensor sequences. Each
                                  sequence is terminated by None.

    @param sequenceLabels         A sequence of string representations of the
                                  current sequence. Each sequence is terminated
                                  by None.

    @param tmLearn:   (bool)      Either False, or True
    @param upLearn:   (None,bool) Either None, False, or True. If None,
                                  union pooler will be skipped.

    @param progressInterval: (int) Prints progress every N iterations,
                                   where N is the value of this param
    """

    currentTime = time.time()

    for i in xrange(len(sensorSequences)):
      sensorPattern = sensorSequences[i]
      sequenceLabel = sequencesLabels[i]

      self.runNetworkOnPattern(sensorPattern,
                               tmLearn=tmLearn,
                               upLearn=upLearn,
                               sequenceLabel=sequenceLabel)

      if progressInterval is not None and i > 0 and i % progressInterval == 0:
        elapsed = (time.time() - currentTime) / 60.0
        print ("Ran {0} / {1} elements of sequence in "
               "{2:0.2f} minutes.".format(i, len(sensorSequences), elapsed))
        currentTime = time.time()
        print MonitorMixinBase.mmPrettyPrintMetrics(
          self.tm.mmGetDefaultMetrics())

    if verbosity >= 2:
      traces = self.tm.mmGetDefaultTraces(verbosity=verbosity)
      print MonitorMixinBase.mmPrettyPrintTraces(traces,
                                                 breakOnResets=
                                                 self.tm.mmGetTraceResets())

      if upLearn is not None:
        traces = self.up.mmGetDefaultTraces(verbosity=verbosity)
        print MonitorMixinBase.mmPrettyPrintTraces(traces,
                                                   breakOnResets=
                                                   self.up.mmGetTraceResets())
      print


  def runNetworkOnPattern(self, sensorPattern, tmLearn=True, upLearn=None,
                          sequenceLabel=None):
    if sensorPattern is None:
      self.tm.reset()
      self.up.reset()
    else:
      self.tm.compute(sensorPattern,
                      formInternalConnections=True,
                      learn=tmLearn,
                      sequenceLabel=sequenceLabel)

      if upLearn is not None:
        activeCells, predActiveCells, burstingCols, = self.getUnionPoolerInput()
        self.up.compute(activeCells,
                        predActiveCells,
                        learn=upLearn,
                        sequenceLabel=sequenceLabel)


  def getUnionPoolerInput(self):
    """
    Gets the Union Pooler input from the Temporal Memory
    """
    activeCells = numpy.zeros(self.tm.numberOfCells()).astype(realDType)
    activeCells[list(self.tm.activeCellsIndices())] = 1

    predictedActiveCells = numpy.zeros(self.tm.numberOfCells()).astype(
      realDType)
    predictedActiveCells[list(self.tm.predictedActiveCellsIndices())] = 1

    burstingColumns = numpy.zeros(self.tm.numberOfColumns()).astype(realDType)
    burstingColumns[list(self.tm.unpredictedActiveColumns)] = 1

    return activeCells, predictedActiveCells, burstingColumns


  def getBurstingColumnsStats(self):
    """
    Gets statistics on the Temporal Memory's bursting columns. Used as a metric
    of Temporal Memory's learning performance.
    :return: mean, standard deviation, and max of Temporal Memory's bursting
    columns over time
    """
    traceData = self.tm.mmGetTraceUnpredictedActiveColumns().data
    resetData = self.tm.mmGetTraceResets().data
    countTrace = []
    for x in xrange(len(traceData)):
      if not resetData[x]:
        countTrace.append(len(traceData[x]))

    mean = numpy.mean(countTrace)
    stdDev = numpy.std(countTrace)
    maximum = max(countTrace)
    return mean, stdDev, maximum
