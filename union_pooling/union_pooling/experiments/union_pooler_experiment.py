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

from nupic.bindings.math import GetNTAReal
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
    TemporalMemoryMonitorMixin)

from sensorimotor.fast_general_temporal_memory import (
    FastGeneralTemporalMemory)
from union_pooling.union_pooler import UnionPooler
from union_pooling.union_pooler_monitor_mixin import UnionPoolerMonitorMixin


class MonitoredGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                     FastGeneralTemporalMemory):
  pass


class MonitoredUnionTemporalPooler(UnionPoolerMonitorMixin, UnionPooler):
  pass



realDType = GetNTAReal()



class UnionPoolerExperiment(object):


  DEFAULT_TEMPORAL_MEMORY_PARAMS = {"columnDimensions": (1024,),
                                    "cellsPerColumn": 8,
                                    "activationThreshold": 20,
                                    "learningRadius": 2048,
                                    "initialPermanence": 0.5,
                                    "connectedPermanence": 0.6,
                                    "minThreshold": 20,
                                    "maxNewSynapseCount": 30,
                                    "permanenceIncrement": 0.10,
                                    "permanenceDecrement": 0.02,
                                    "seed": 42}


  DEFAULT_UNION_POOLER_PARAMS = {"columnDimensions": [1024],
                                 "potentialPct": 0.5,
                                 "globalInhibition": True,
                                 "localAreaDensity": -1,
                                 "numActiveColumnsPerInhArea": 20,
                                 "stimulusThreshold": 0,
                                 "synPermInactiveDec": 0.01,
                                 "synPermActiveInc": 0.1,
                                 "synPermConnected": 0.1,
                                 "minPctOverlapDutyCycle": 0.001,
                                 "minPctActiveDutyCycle": 0.001,
                                 "dutyCyclePeriod": 1000,
                                 "maxBoost": 10.0,
                                 "seed": 42,
                                 "spVerbosity": 0,
                                 "wrapAround": True}


  def __init__(self, tmOverrides=None, upOverrides=None, seed=42):
    # Initialize Temporal Memory
    params = dict(self.DEFAULT_TEMPORAL_MEMORY_PARAMS)
    params.update(tmOverrides or {})
    params["seed"] = seed
    self.tm = MonitoredGeneralTemporalMemory(mmName="TM", **params)

    # Initialize Union Pooler layer
    params = dict(self.DEFAULT_UNION_POOLER_PARAMS)
    params["inputDimensions"] = [self.tm.numberOfCells()]
    params["potentialRadius"] = self.tm.numberOfCells()
    params["seed"] = seed
    params.update(upOverrides or {})

    # TODO Implement MonitoredUnionPooler
    self.up = UnionPooler(**params)


  def runNetworkOnSequence(self, sensorSequence, sequenceLabels, tmLearn=True,
                           upLearn=None, verbosity=0, progressInterval=None):
    """
    Runs Union Pooler network on specified sequence.

    @param sensorSequence         TODO
    @param sequenceLabels         TODO
    @param tmLearn:   (bool)      Either False, or True
    @param upLearn:   (None,bool) Either None, False, or True. If None,
                                  union pooler will be skipped.

    @param progressInterval: (int) Prints progress every N iterations,
                                   where N is the value of this param
    """

    currentTime = time.time()

    for i in xrange(len(sensorSequence)):
      sensorPattern = sensorSequence[i]
      sequenceLabel = sequenceLabels[i]

      self.runNetworkOnPattern(sensorPattern, tmLearn=tmLearn, upLearn=upLearn,
                               sequenceLabel=sequenceLabel)

      if progressInterval is not None and i > 0 and i % progressInterval == 0:
        print ("Ran {0} / {1} elements of sequence in "
               "{2:0.2f} seconds.".format(i, len(sensorSequence),
                                          time.time() - currentTime))
        currentTime = time.time()

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
      # TODO implement reset() in monitor
      # self.up.reset()
    else:
      self.tm.compute(sensorPattern,
                      formInternalConnections=True,
                      learn=tmLearn,
                      sequenceLabel=sequenceLabel)

      if upLearn is not None:
        activeCells, predictedActiveCells, burstingColumns, = (
            self.getUnionPoolerInput())
        activeColumnsArray = numpy.zeros(self.up.getNumColumns())

        self.up.compute(activeCells,
                        upLearn,
                        activeColumnsArray,
                        burstingColumns,
                        predictedActiveCells,
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
