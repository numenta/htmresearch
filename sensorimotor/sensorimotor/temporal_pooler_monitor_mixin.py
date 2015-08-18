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
Temporal Pooler mixin that enables detailed monitoring of history.
"""

from collections import defaultdict

import numpy

from nupic.research.monitor_mixin.metric import Metric
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.plot import Plot
from nupic.research.monitor_mixin.trace import (
  IndicesTrace, StringsTrace, BoolsTrace, MetricsTrace)



class TemporalPoolerMonitorMixin(MonitorMixinBase):
  """
  Mixin for TemporalPooler that stores a detailed history, for inspection and
  debugging.
  """

  def __init__(self, *args, **kwargs):
    super(TemporalPoolerMonitorMixin, self).__init__(*args, **kwargs)

    self._mmResetActive = True  # First iteration is always a reset


  def mmGetTraceActiveCells(self):
    """
    @return (Trace) Trace of active cells
    """
    return self._mmTraces["activeCells"]


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


  def mmGetDataOverlap(self):
    """
    Returns 2D matrix of overlaps for sets of active cells between pairs of
    iterations. Both the rows and columns are iterations, and the values in the
    matrix are the size of overlaps between sets of active cells for those
    iterations.

    @return (numpy.array) Overlap data
    """
    self._mmComputeSequenceRepresentationData()
    return self._mmData["overlap"]


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


  def mmGetPlotConnectionsPerColumn(self, title=None):
    """
    Returns plot of # connections per column.

    @return (Plot) plot
    """
    plot = Plot(self, title)
    plot.addGraph(sorted(self._connectedCounts.tolist(), reverse=True),
                  position=211,
                  xlabel="column", ylabel="# connections")
    plot.addHistogram(self._connectedCounts.tolist(),
                      position=212,
                      bins=len(self._connectedCounts) / 10,
                      xlabel="# connections", ylabel="# columns")
    return plot


  def mmGetCellActivityPlot(self, title=None, showReset=False,
                            resetShading=0.25):
    """ Returns plot of the cell activity.
    @param title an optional title for the figure
    @param showReset if true, the first set of cell activities after a reset
                        will have a gray background
    @param resetShading If showReset is true, this float specifies the
    intensity of the reset background with 0.0 being white and 1.0 being black
    @return (Plot) plot
    """
    cellTrace = self._mmTraces["activeCells"].data
    cellCount = self.getNumColumns()
    activityType = "Cell Activity"
    return self.mmGetCellTracePlot(cellTrace, cellCount, activityType,
                                   title=title, showReset=showReset,
                                   resetShading=resetShading)


  def mmGetPermanencesPlot(self, title=None):
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

  def _mmComputeSequenceRepresentationData(self):
    if not self._sequenceRepresentationDataStale:
      return

    activeCellsTrace = self.mmGetTraceActiveCells()
    sequenceLabelsTrace = self.mmGetTraceSequenceLabels()
    resetsTrace = self.mmGetTraceResets()

    n = len(activeCellsTrace.data)
    overlap = numpy.empty((n, n), dtype=int)
    stabilityConfusion = []
    distinctnessConfusion = []

    for i in xrange(n):
      for j in xrange(i+1):
        numActiveCells = len(activeCellsTrace.data[i])
        numOverlap = len(activeCellsTrace.data[i] & activeCellsTrace.data[j])
        overlap[i][j] = numOverlap
        overlap[j][i] = numOverlap

        if (i != j and
            sequenceLabelsTrace.data[i] is not None and
            not resetsTrace.data[i] and
            sequenceLabelsTrace.data[j] is not None and
            not resetsTrace.data[j]):
          if sequenceLabelsTrace.data[i] == sequenceLabelsTrace.data[j]:
            stabilityConfusion.append(numActiveCells - numOverlap)
          else:
            distinctnessConfusion.append(numOverlap)

    self._mmData["overlap"] = overlap
    self._mmData["stabilityConfusion"] = stabilityConfusion
    self._mmData["distinctnessConfusion"] = distinctnessConfusion

    self._sequenceRepresentationDataStale = False


  # ==============================
  # Overrides
  # ==============================

  def compute(self, *args, **kwargs):
    sequenceLabel = None
    if "sequenceLabel" in kwargs:
      sequenceLabel = kwargs["sequenceLabel"]
      del kwargs["sequenceLabel"]

    activeColumns = super(TemporalPoolerMonitorMixin, self).compute(*args,
                                                                    **kwargs)
    activeColumns = set(activeColumns)
    activeCells = activeColumns  # TODO: Update when moving to a cellular TP

    self._mmTraces["activeCells"].data.append(activeCells)
    self._mmTraces["sequenceLabels"].data.append(sequenceLabel)

    self._mmTraces["resets"].data.append(self._mmResetActive)
    self._mmResetActive = False

    self._mmTraces["connectionsPerColumnMetric"].data.append(
      Metric(self, "connections per column", self._connectedCounts.tolist()))

    self._sequenceRepresentationDataStale = True


  def reset(self):
    super(TemporalPoolerMonitorMixin, self).reset()

    self._mmResetActive = True


  def mmGetDefaultTraces(self, verbosity=1):
    traces = [self.mmGetTraceActiveCells()]

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
    super(TemporalPoolerMonitorMixin, self).mmClearHistory()

    self._mmTraces["activeCells"] = IndicesTrace(self, "active cells")
    self._mmTraces["sequenceLabels"] = StringsTrace(self, "sequence labels")
    self._mmTraces["resets"] = BoolsTrace(self, "resets")
    self._mmTraces["connectionsPerColumnMetric"] = MetricsTrace(
      self, "connections per column (metric)")

    self._sequenceRepresentationDataStale = True
