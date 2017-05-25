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
Union Temporal Pooler mixin that enables detailed monitoring of history.
"""

from collections import defaultdict

import numpy

from nupic.algorithms.monitor_mixin.metric import Metric
from nupic.algorithms.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from htmresearch.algorithms.union_temporal_pooler import UnionTemporalPooler
from nupic.algorithms.monitor_mixin.plot import Plot
from nupic.algorithms.monitor_mixin.trace import (
  IndicesTrace, StringsTrace, BoolsTrace, MetricsTrace, CountsTrace)

from nupic.bindings.math import GetNTAReal


realDType = GetNTAReal()
uintType = "uint32"



class UnionTemporalPoolerMonitorMixin(MonitorMixinBase):
  """
  Mixin for UnionTemporalPooler that stores a detailed history, for inspection and
  debugging.
  """

  def __init__(self, *args, **kwargs):
    super(UnionTemporalPoolerMonitorMixin, self).__init__(*args, **kwargs)

    self._mmResetActive = True  # First iteration is always a reset
    self.mmClearHistory()


  def mmGetDataUnionSDRDutyCycle(self):
    """
    @return (list) duty cycles for union SDR bits
    """
    return self._mmData["unionSDRDutyCycle"].tolist()


  def mmGetMetricUnionSDRDutyCycle(self):
    """
    @return (Metric) duty cycle metric for union SDR bits
    """
    data = self.mmGetDataUnionSDRDutyCycle()
    return Metric(self, "Union SDR duty cycle", data)


  def mmGetPlotUnionSDRDutyCycle(self, title="Union SDR Duty Cycle"):
    """
    @return (Plot) Plot of union SDR duty cycle.
    """
    plot = Plot(self, title)
    unionSDRDutyCycle = self.mmGetDataUnionSDRDutyCycle()
    plot.addGraph(sorted(unionSDRDutyCycle, reverse=True),
                  position=211,
                  xlabel="Union SDR Bit", ylabel="Duty Cycle")
    plot.addHistogram(unionSDRDutyCycle,
                      position=212,
                      bins=len(unionSDRDutyCycle) / 10,
                      xlabel="Duty Cycle", ylabel="# Union SDR Bits")
    return plot


  def mmGetDataPersistenceDutyCycle(self):
    """
    @return (list) duty cycles for persistences
    """
    return self._mmData["persistenceDutyCycle"].tolist()


  def mmGetMetricPersistenceDutyCycle(self):
    """
    @return (Metric) duty cycle metric for persistences
    """
    data = self.mmGetDataPersistenceDutyCycle()
    return Metric(self, "Persistence duty cycle", data)


  def mmGetPlotPersistenceDutyCycle(self, title="Persistence Duty Cycle"):
    """
    @return (Plot) Plot of persistence duty cycle.
    """
    plot = Plot(self, title)
    persistenceDutyCycle = self.mmGetDataPersistenceDutyCycle()
    plot.addGraph(sorted(persistenceDutyCycle, reverse=True),
                  position=211,
                  xlabel="Union SDR Bit", ylabel="Persistence Duty Cycle")
    plot.addHistogram(persistenceDutyCycle,
                      position=212,
                      bins=len(persistenceDutyCycle) / 10,
                      xlabel="Persistence Duty Cycle", ylabel="# Union SDR Bits")
    return plot


  def mmGetDataBitlife(self):
    """
    The bitlife is the number of time steps for which a union SDR bit remains
    active. This method returns a list of all bitlife values with each entry
    corresponding to an instance in which a bit turns on.

    @return (list) All bitlife values
    """
    return self._mmComputeBitLifeStats()


  def mmGetMetricBitlife(self):
    """
    See `mmGetDataBitlife` for description of bitlife.

    @return (Metric) bitlife metric
    """
    data = self._mmComputeBitLifeStats()
    return Metric(self, "Union SDR bitlife", data)


  def mmGetPlotBitlife(self, title="Bitlife Statistics"):
    """
    @return (Plot) Plot of bitlife statistics.
    """
    plot = Plot(self, title)
    bitlife = self.mmGetDataBitlife()
    print bitlife
    plot.addGraph(sorted(bitlife, reverse=True),
                  position=211,
                  xlabel="Union SDR Bit", ylabel="Bitlife")
    plot.addHistogram(bitlife,
                      position=212,
                      bins=max(len(bitlife) / 10, 3),
                      xlabel="Bitlife", ylabel="# Union SDR Bits")
    return plot


  def mmGetDataConnectedCounts(self):
    connectedCounts = numpy.zeros(self.getNumColumns(), dtype='uint32')
    self.getConnectedCounts(connectedCounts)
    return connectedCounts


  def mmGetMetricConnectedCounts(self):
    data = self.mmGetDataConnectedCounts()
    return Metric(self, "Connected synapse counts", data)


  def mmGetTraceUnionSDR(self):
    """
    @return (Trace) Trace of union SDR
    """
    return self._mmTraces["unionSDR"]


  def mmGetPlotUnionSDRActivity(self, title="Union SDR Activity Raster",
                                showReset=False, resetShading=0.25):
    """ Returns plot of the activity of union SDR bits.
    @param title an optional title for the figure
    @param showReset if true, the first set of activities after a reset
                        will have a gray background
    @param resetShading If showReset is true, this float specifies the
    intensity of the reset background with 0.0 being white and 1.0 being black
    @return (Plot) plot
    """
    unionSDRTrace = self.mmGetTraceUnionSDR().data
    columnCount = self.getNumColumns()
    activityType = "Union SDR Activity"
    return self.mmGetCellTracePlot(unionSDRTrace, columnCount, activityType,
                                   title=title, showReset=showReset,
                                   resetShading=resetShading)


  def mmGetTraceSequenceLabels(self):
    """
    @return (Trace) Trace of sequence labels
    """
    return self._mmTraces["sequenceLabels"]


  def mmGetTraceResets(self):
    """
    @return (Trace) Trace of resets
    """
    return self._mmTraces["resets"]


  def mmGetTraceConnectionsPerColumnMetric(self):
    """
    @return (Trace) Trace of connections per column metric
    """
    return self._mmTraces["connectionsPerColumnMetric"]


  def mmGetPlotConnectionsPerColumn(self, title="Connections per Columns"):
    """
    Returns plot of # connections per column.

    @return (Plot) plot
    """
    plot = Plot(self, title)
    connectedCounts = numpy.ndarray(self.getNumColumns(), dtype=uintType)
    self.getConnectedCounts(connectedCounts)

    plot.addGraph(sorted(connectedCounts.tolist(), reverse=True),
                  position=211,
                  xlabel="column", ylabel="# connections")

    plot.addHistogram(connectedCounts.tolist(),
                      position=212,
                      bins=len(connectedCounts) / 10,
                      xlabel="# connections", ylabel="# columns")
    return plot


  def mmGetDataOverlap(self):
    """
    Returns 2D matrix of overlaps for sets of active bits between pairs of
    iterations. Both the rows and columns are iterations, and the values in the
    matrix are the value of the overlap metric between sets of active bits for
    those iterations.

    @return (numpy.array) Overlap data
    """
    self._mmComputeSequenceRepresentationData()
    return self._mmData["overlap"]


  def mmPrettyPrintDataOverlap(self):
    """
    Returns pretty-printed string representation of overlap metric data.
    (See `mmGetDataOverlap`.)

    @return (string) Pretty-printed data
    """
    matrix = self.mmGetDataOverlap()
    resetsTrace = self.mmGetTraceResets()

    text = ""

    for i, row in enumerate(matrix):
      if resetsTrace.data[i]:
        text += "\n"

      for j, item in enumerate(row):
        if resetsTrace.data[j]:
          text += "    "

        text += "{:4}".format(item)

      text += "\n"

    return text


  def mmGetMetricStabilityConfusion(self):
    """
    For each iteration that doesn't follow a reset, looks at every other
    iteration for the same world that doesn't follow a reset, and computes the
    number of bits that show up in one or the other set of active cells for
    that iteration, but not both. This metric returns the distribution of those
    numbers.

    @return (Metric) Stability confusion metric
    """
    self._mmComputeSequenceRepresentationData()
    numbers = self._mmData["stabilityConfusion"]
    return Metric(self, "stability confusion", numbers)


  def mmGetPlotStability(self, title="Stability", showReset=False,
                         resetShading=0.25):
    """
    Returns plot of the overlap metric between union SDRs within a sequence.
    @param title an optional title for the figure
    @return (Plot) plot
    """
    plot = Plot(self, title)
    self._mmComputeSequenceRepresentationData()
    data = self._mmData["stabilityConfusion"]
    plot.addGraph(sorted(data, reverse=True),
                  position=211,
                  xlabel="Time steps", ylabel="Overlap")
    plot.addHistogram(data,
                      position=212,
                      bins=100,
                      xlabel="Overlap", ylabel="# time steps")
    return plot


  def mmGetMetricDistinctnessConfusion(self):
    """
    For each iteration that doesn't follow a reset, looks at every other
    iteration for every other world that doesn't follow a reset, and computes
    the number of bits that show up in both sets of active cells those that
    iteration. This metric returns the distribution of those numbers.

    @return (Metric) Distinctness confusion metric
    """
    self._mmComputeSequenceRepresentationData()
    numbers = self._mmData["distinctnessConfusion"]
    return Metric(self, "distinctness confusion", numbers)


  def mmGetPlotDistinctness(self, title="Distinctiness", showReset=False,
                            resetShading=0.25):
    """
    Returns plot of the overlap metric between union SDRs between sequences.
    @param title an optional title for the figure
    @return (Plot) plot
    """
    plot = Plot(self, title)
    self._mmComputeSequenceRepresentationData()
    data = self._mmData["distinctnessConfusion"]
    plot.addGraph(sorted(data, reverse=True),
                  position=211,
                  xlabel="Time steps", ylabel="Overlap")
    plot.addHistogram(data,
                      position=212,
                      bins=100,
                      xlabel="Overlap", ylabel="# time steps")
    return plot


  def mmGetPlotPermanences(self, title="Permanences"):
    """ Returns plot of column permanences.
    @param title an optional title for the figure
    @return (Plot) plot
    """
    plot = Plot(self, title)

    data = numpy.zeros((self.getNumColumns(), self.getNumInputs()))
    for i in xrange(self.getNumColumns()):
      self.getPermanence(i, data[i])

    plot.add2DArray(data, xlabel="Permanences", ylabel="Column")
    return plot


  def mmPrettyPrintDataOverlap(self):
    """
    Returns pretty-printed string representation of overlap data.
    (See `mmGetDataOverlap`.)

    @return (string) Pretty-printed data
    """
    matrix = self.mmGetDataOverlap()
    resetsTrace = self.mmGetTraceResets()

    text = ""

    for i, row in enumerate(matrix):
      if resetsTrace.data[i]:
        text += "\n"

      for j, item in enumerate(row):
        if resetsTrace.data[j]:
          text += "    "

        text += "{:4}".format(item)

      text += "\n"

    return text


  # ==============================
  # Helpers
  # ==============================


  @staticmethod
  def _mmUpdateDutyCyclesHelper(dutyCycles, newInput, period):
    """
    Updates a duty cycle estimate with a new value. This is a helper
    function that is used to update several duty cycle variables in
    the Column class, such as: overlapDutyCucle, activeDutyCycle,
    minPctDutyCycleBeforeInh, minPctDutyCycleAfterInh, etc. returns
    the updated duty cycle. Duty cycles are updated according to the following
    formula:

                  (period - 1)*dutyCycle + newValue
      dutyCycle := ----------------------------------
                              period

    Parameters:
    ----------------------------
    @param dutyCycles: An array containing one or more duty cycle values that need
                    to be updated
    @param newInput: A new numerical value used to update the duty cycle
    @param period:  The period of the duty cycle
    """
    assert(period >= 1)
    return (dutyCycles * (period -1.0) + newInput) / period


  def _mmUpdateDutyCycles(self):
    """
    Update the duty cycle variables internally tracked by the TM mixin.
    """
    period = self.getDutyCyclePeriod()

    unionSDRArray = numpy.zeros(self.getNumColumns())
    unionSDRArray[list(self._mmTraces["unionSDR"].data[-1])] = 1

    self._mmData["unionSDRDutyCycle"] = \
      UnionTemporalPoolerMonitorMixin._mmUpdateDutyCyclesHelper(
        self._mmData["unionSDRDutyCycle"], unionSDRArray, period)

    self._mmData["persistenceDutyCycle"] = \
      UnionTemporalPoolerMonitorMixin._mmUpdateDutyCyclesHelper(
        self._mmData["persistenceDutyCycle"], self._poolingActivation, period)


  def _mmComputeSequenceRepresentationData(self):
    """
    Calculates values for the overlap distance matrix, stability within a
    sequence, and distinctness between sequences. These values are cached so
    that they do need to be recomputed for calls to each of several accessor
    methods that use these values.
    """
    if not self._sequenceRepresentationDataStale:
      return

    unionSDRTrace = self.mmGetTraceUnionSDR()
    sequenceLabelsTrace = self.mmGetTraceSequenceLabels()
    resetsTrace = self.mmGetTraceResets()

    n = len(unionSDRTrace.data)
    overlapMatrix = numpy.empty((n, n), dtype=uintType)
    stabilityConfusionUnionSDR = []
    distinctnessConfusionUnionSDR = []

    for i in xrange(n):
      for j in xrange(i+1):
        overlapUnionSDR = len(unionSDRTrace.data[i] & unionSDRTrace.data[j])

        overlapMatrix[i][j] = overlapUnionSDR
        overlapMatrix[j][i] = overlapUnionSDR

        if (i != j and
            sequenceLabelsTrace.data[i] is not None and
            not resetsTrace.data[i] and
            sequenceLabelsTrace.data[j] is not None and
            not resetsTrace.data[j]):
          if sequenceLabelsTrace.data[i] == sequenceLabelsTrace.data[j]:
            stabilityConfusionUnionSDR.append(overlapUnionSDR)
          else:
            distinctnessConfusionUnionSDR.append(overlapUnionSDR)

    self._mmData["overlap"] = overlapMatrix
    self._mmData["stabilityConfusion"] = stabilityConfusionUnionSDR
    self._mmData["distinctnessConfusion"] = distinctnessConfusionUnionSDR

    self._sequenceRepresentationDataStale = False


  def _mmComputeBitLifeStats(self):
    """
    @return (list) Life duration of all active bits
    """
    bitLifeList = []
    traceData = self._mmTraces["unionSDR"].data
    n = len(traceData)
    bitLifeCounter = numpy.zeros(self.getNumColumns())
    preActiveCells = set()
    for t in xrange(n-1):
      preActiveCells = set(numpy.where(bitLifeCounter>0)[0])
      newActiveCells = list(traceData[t] - preActiveCells)
      stopActiveCells = list(preActiveCells - traceData[t])
      continuousActiveCells = list(preActiveCells & traceData[t] )
      bitLifeList += list(bitLifeCounter[stopActiveCells])

      bitLifeCounter[stopActiveCells] = 0
      bitLifeCounter[newActiveCells] = 1
      bitLifeCounter[continuousActiveCells] += 1

    return bitLifeList




  # ==============================
  # Overrides
  # ==============================

  def compute(self, *args, **kwargs):
    sequenceLabel = kwargs.pop("sequenceLabel", None)

    unionSDR = super(UnionTemporalPoolerMonitorMixin, self).compute(*args,
                                                                    **kwargs)

    ### From spatial pooler
    # total number of connections
    connectedCounts = numpy.zeros(self.getNumColumns(), dtype=uintType)
    self.getConnectedCounts(connectedCounts)
    numConnections = numpy.sum(connectedCounts)

    self._mmTraces["unionSDR"].data.append(set(unionSDR))
    self._mmTraces["numConnections"].data.append(numConnections)
    self._mmTraces["sequenceLabels"].data.append(sequenceLabel)
    self._mmTraces["resets"].data.append(self._mmResetActive)
    self._mmResetActive = False
    self._mmTraces["connectionsPerColumnMetric"].data.append(
      Metric(self, "connections per column", self._connectedCounts.tolist()))

    self._sequenceRepresentationDataStale = True
    self._mmUpdateDutyCycles()


  def reset(self):
    super(UnionTemporalPoolerMonitorMixin, self).reset()

    self._mmResetActive = True


  def mmGetDefaultTraces(self, verbosity=1):
    traces = [self.mmGetTraceUnionSDR()]

    if verbosity == 1:
      traces = [trace.makeCountsTrace() for trace in traces]

    return traces + [
      self.mmGetTraceConnectionsPerColumnMetric(),
      self.mmGetTraceSequenceLabels()
    ]


  def mmGetDefaultMetrics(self, verbosity=1):
    metrics = ([Metric.createFromTrace(trace)
                for trace in self.mmGetDefaultTraces()[:-2]])

    connectionsPerColumnMetricIntial = (
      self._mmTraces["connectionsPerColumnMetric"].data[0].copy())
    connectionsPerColumnMetricIntial.title += " (initial)"
    connectionsPerColumnMetricFinal = (
      self._mmTraces["connectionsPerColumnMetric"].data[-1].copy())
    connectionsPerColumnMetricFinal.title += " (final)"

    metrics += [self.mmGetMetricStabilityConfusion(),
                self.mmGetMetricDistinctnessConfusion(),
                connectionsPerColumnMetricIntial,
                connectionsPerColumnMetricFinal]

    return metrics


  def mmClearHistory(self):
    super(UnionTemporalPoolerMonitorMixin, self).mmClearHistory()

    self._mmTraces["unionSDR"] = IndicesTrace(self, "union SDR")
    self._mmTraces["sequenceLabels"] = StringsTrace(self, "sequence labels")
    self._mmTraces["resets"] = BoolsTrace(self, "resets")
    self._mmTraces["connectionsPerColumnMetric"] = MetricsTrace(
      self, "connections per column (metric)")

    self._mmData["unionSDRDutyCycle"] = numpy.zeros(self.getNumColumns(), dtype=realDType)
    self._mmData["persistenceDutyCycle"] = numpy.zeros(self.getNumColumns(), dtype=realDType)

    self._mmTraces["numConnections"] = CountsTrace(self, "connections")

    self._sequenceRepresentationDataStale = True

