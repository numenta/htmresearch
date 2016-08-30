# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
Spatial Pooler mixin that enables detailed monitoring of history.
"""
import copy
from collections import defaultdict
import numpy
from nupic.research.monitor_mixin.metric import Metric
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.trace import (
  IndicesTrace, CountsTrace, BoolsTrace, MetricsTrace)
from nupic.bindings.math import GetNTAReal

realDType = GetNTAReal()
uintType = "uint32"



class SpatialPoolerMonitorMixin(MonitorMixinBase):
  """
  Mixin for SpatialPooler that stores a detailed history, for inspection and
  debugging.
  """

  def mmGetTraceActiveColumns(self):
    """
    @return (Trace) Trace of active columns
    """
    return self._mmTraces["activeColumns"]


  def mmGetTraceActiveInputs(self):
    """
    @return (Trace) Trace of active inputs
    """
    return self._mmTraces["activeInputs"]


  def mmGetTraceNumConnections(self):
    """
    @return (Trace) Trace of # connections
    """
    return self._mmTraces["numConnections"]


  def mmGetTraceLearn(self):
    """
    @return (Trace) Trace of learn flags
    """
    return self._mmTraces["learn"]


  def mmGetDataInitialPermanence(self):
    """
    @return (array) Initial permamnences
    """
    return self._mmData["initialPermanence"]


  def mmGetDataDutyCycles(self):
    """
    @return (list) duty cycles for all columns
    """
    dutyCycles = [0] * self._numColumns
    activeColumns = self.mmGetTraceActiveColumns().data
    for i in range(len(activeColumns)):
        for j in range(len(activeColumns[i])):
            dutyCycles[activeColumns[i][j]] += 1

    return dutyCycles


  def mmGetMetricDutyCycles(self):
    """
    @return (Metric) duty cycle metric
    """
    dutyCycles = self.mmGetDataDutyCycles()

    return Metric(self,
                  "total column duty cycles",
                  dutyCycles)


  def mmGetMetricEntropy(self):
    """
    @return (Metric) entropy 
    """
    dutyCycles = self.mmGetDataDutyCycles()
    MIN_ACTIVATION_PROB = 0.000001

    dutyCycles[dutyCycles < MIN_ACTIVATION_PROB] = MIN_ACTIVATION_PROB
    dutyCycles = dutyCycles / numpy.sum(dutyCycles)

    entropy = -numpy.dot(dutyCycles, numpy.log2(dutyCycles))
    return Metric(self, "entropy", entropy)
  
  # ==============================
  # Overrides
  # ==============================

  def compute(self, inputVector, learn, activeArray, *args, **kwargs):
    super(SpatialPoolerMonitorMixin, self).compute(
        inputVector, learn, activeArray, *args,**kwargs)

    # extract indices of active columns
    activeColumns = numpy.nonzero(activeArray)

    # total number of connections
    connectedCounts = numpy.zeros(self._numColumns, dtype=uintType)
    self.getConnectedCounts(connectedCounts)
    numConnections = numpy.sum(connectedCounts)
        

    self._mmTraces["activeInputs"].data.append(numpy.where(inputVector)[0])
    self._mmTraces["activeColumns"].data.append(activeColumns)
    self._mmTraces["numConnections"].data.append(numConnections)
    self._mmTraces["learn"].data.append(learn)    


  def mmGetDefaultTraces(self, verbosity=1):
    traces = [
      self.mmGetTraceActiveColumns(),
    ]

    if verbosity <= 1:
      traces = [trace.makeCountsTrace() for trace in traces]

    traces += [self.mmGetTraceNumConnections()]

    return traces


  def mmGetDefaultMetrics(self, verbosity=1):
    metrics = ([Metric.createFromTrace(trace)
                for trace in self.mmGetDefaultTraces()])
    metrics += [self.mmGetMetricDutyCycles()]

    return metrics


  def mmClearHistory(self):
    super(SpatialPoolerMonitorMixin, self).mmClearHistory()

    self._mmTraces["activeColumns"] = IndicesTrace(self, "active columns")
    self._mmTraces["activeInputs"] = IndicesTrace(self, "active inputs")
    self._mmTraces["numConnections"] = CountsTrace(self, "connections")
    self._mmTraces["learn"] = BoolsTrace(self, "learn")    

    initialPermanence = numpy.zeros((self._numColumns, self._numInputs),
                                    dtype=realDType)
    for c in range(self._numColumns):
      self.getPermanence(c, initialPermanence[c, :])

    self._mmData["initialPermanence"] = initialPermanence



  def recoverPermanence(self, columnIndex):
    """
    Recover permamnece for a single column
    :param columnIndex: index of the column of interest
    :return: permInfo (dict) with the following traces over time
              avgPermConnectedSyn: mean permanence for connected synapses
              avgPermNonConnectedSyn: mean permanence for non-connected synapses
              numConnectedSyn: number of connected synapses
              numNonConnectedSyn: number of non-connected synapses
    """
    activeColumns = self.mmGetTraceActiveColumns()
    activeInputs = self.mmGetTraceActiveInputs()
    learnTrace = self.mmGetTraceLearn()
    initialPermanence = self.mmGetDataInitialPermanence()

    (columnNumber, inputSize) = initialPermanence.shape

    # potential mask for the column of interest
    potential = numpy.zeros((inputSize), dtype=uintType)
    self.getPotential(columnIndex, potential)
    maskPotential = numpy.where(potential > 0)[0]

    permChanges = numpy.zeros(inputSize, dtype=realDType)
    perm = numpy.copy(initialPermanence[columnIndex, :])

    numStep = len(activeColumns.data)
    numConnectedSyn = numpy.zeros((numStep))
    avgPermConnected = numpy.zeros((numStep))
    avgPermNonConnected = numpy.zeros((numStep))

    for i in range(numStep):
      if learnTrace.data[i] is False:
        continue

      if columnIndex in activeColumns.data[i][0]:
        # print "Column {} active at step {}".format(columnIndex, i)
        # the logic here matches _adaptSynapses funciton in SP
        permChanges.fill(-1 * self._synPermInactiveDec)
        permChanges[activeInputs.data[i]] = self._synPermActiveInc
        perm[maskPotential] += permChanges[maskPotential]

        perm[perm < self._synPermTrimThreshold] = 0
        numpy.clip(perm, self._synPermMin, self._synPermMax, out=perm)

      numConnectedSyn[i] = numpy.sum(perm > self._synPermConnected)
      permMask = perm[maskPotential]
      avgPermConnected[i] = numpy.mean(permMask[
                                         permMask > self._synPermConnected])
      avgPermNonConnected[i] = numpy.mean(permMask[
                                            permMask < self._synPermConnected])

    truePermanence = numpy.zeros((inputSize), dtype=realDType)
    self.getPermanence(columnIndex, truePermanence)
    if numpy.max(numpy.abs(perm - truePermanence)) > 1e-10:
      raise RuntimeError("Permamnence reconstruction failed. The reconstructed"
                         "permamnence does not match the real permamnence")

    numNonConnectedSyn = len(maskPotential) - numConnectedSyn
    permInfo = {"avgPermConnectedSyn": avgPermConnected,
                "avgPermNonConnectedSyn": avgPermNonConnected,
                "numConnectedSyn": numConnectedSyn,
                "numNonConnectedSyn": numNonConnectedSyn}

    return permInfo