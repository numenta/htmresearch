# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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

"""
Temporal Pooler mixin that enables detailed monitoring of history.
"""

from collections import defaultdict

from nupic.research.monitor_mixin.metric import Metric
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.plot import HistogramPlot
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


  def mmGetDataStabilityConfusion(self):
    """
    Returns data representing the stability of sequence representations. This is
    measured by checking, for a given sequence, whether the active cells at any
    iteration along the sequence are same as or different from the active cells
    at each other iteration along the sequence.

    The dictionary returned is a mapping:
    sequence label (string) => confusion matrix (list)

    For the confusion matrix, the rows are iterations, and the columns are the
    same iterations. Each entry represents the number of active cells that
    show up in either of the two iterations but not both (symmetric difference).

    @return (dict) Stability confusion data
    """
    self._mmComputeSequenceRepresentationData()
    return self._mmData["stabilityConfusion"]


  def mmGetDataDistinctnessConfusion(self):
    """
    Returns data representing the distinctness of sequence representations.
    This is measured by comparing each sequence representation (the union of
    active cells representing the sequence) against every other sequence
    representation, and checking whether the representations are distinct.

    The tuple returned contains the following values:
    (confusion matrix, labels)

    For the confusion matrix, the rows are sequence representations, and the
    columns are the same sequence representations. Each entry represents the
    number of active cells that show up both of the sequence representations
    (intersection).

    The labels are the sequence labels for the rows / columns (useful for
    pretty-printing this data).

    NOTE: This data is only really relevant when the representations are
    fairly stable. If the stability is low, then the union of active cells
    representing the sequence will be a large set, and this data won't mean
    much.

    @return (dict) Distinctness confusion data
    """
    self._mmComputeSequenceRepresentationData()
    return (self._mmData["distinctnessConfusionMatrix"],
            self._mmData["distinctnessConfusionLabels"])


  def mmGetMetricStabilityConfusion(self):
    """
    Returns metric made of values from the stability confusion matrix.
    (See `mmGetDataStabilityConfusion`.)

    @return (Metric) Stability confusion metric
    """
    data = self.mmGetDataStabilityConfusion()
    numbers = []

    for matrix in data.values():
      for i in xrange(len(matrix)):
        for j in xrange(len(matrix[i])):
          if i != j:  # Ignoring diagonal
            numbers.append(matrix[i][j])

    return Metric(self, "stability confusion", numbers)


  def mmGetMetricDistinctnessConfusion(self):
    """
    Returns metric made of values from the distinctness confusion matrix
    (excluding diagonals). (See `mmGetDataDistinctnessConfusion`.)

    @return (Metric) Distinctness confusion metric
    """
    data, _ = self.mmGetDataDistinctnessConfusion()
    numbers = []

    for i in xrange(len(data)):
      for j in xrange(len(data[i])):
        if i != j:  # Ignoring diagonal
          numbers.append(data[i][j])

    return Metric(self, "distinctness confusion", numbers)


  def mmGetPlotConnectionsPerColumn(self, label=None):
    """
    Returns plot of # connections per column.

    @return (HistogramPlot) plot
    """
    return HistogramPlot(self, "# connections per column",
                         self._connectedCounts.tolist(),
                         label=label,
                         xlabel="# connections", ylabel="# columns",
                         bucketSize=len(self._connectedCounts) / 10)


  def mmPrettyPrintDataStabilityConfusion(self):
    """
    Returns pretty-printed string representation of stability confusion data.
    (See `mmGetDataStabilityConfusion`.)

    @return (string) Pretty-printed data
    """
    data = self.mmGetDataStabilityConfusion()
    text = ""

    for sequenceLabel, confusionMatrix in data.iteritems():
      text += "{0}:\n\n".format(sequenceLabel)
      text += ("\n".join([''.join(['{:4}'.format(item) for item in row])
                          for row in confusionMatrix]))
      text += "\n\n"

    return text


  def mmPrettyPrintDataDistinctnessConfusion(self):
    """
    Returns pretty-printed string representation of distinctness confusion data.
    (See `mmGetDataDistinctnessConfusion`.)

    @return (string) Pretty-printed data
    """
    matrix, labels = self.mmGetDataDistinctnessConfusion()
    labelText = ", ".join(labels)

    text = ""
    text += "(rows: {0})\n".format(labelText)
    text += "(cols: {0})\n\n".format(labelText)
    text += ("\n".join([''.join(['{:4}'.format(item) for item in row])
                        for row in matrix]))
    text += "\n"

    return text


  # ==============================
  # Helpers
  # ==============================

  def _mmComputeSequenceRepresentationData(self):
    if not self._sequenceRepresentationDataStale:
      return

    self._mmData["activeCellsListForSequence"] = defaultdict(list)
    self._mmData["activeCellsUnionForSequence"] = defaultdict(set)
    self._mmData["stabilityConfusion"] = {}
    self._mmData["distinctnessConfusionMatrix"] = []
    self._mmData["distinctnessConfusionLabels"] = []

    activeCellsTrace = self.mmGetTraceActiveCells()
    sequenceLabelsTrace = self.mmGetTraceSequenceLabels()
    resetsTrace = self.mmGetTraceResets()

    for i, activeCells in enumerate(activeCellsTrace.data):
      sequenceLabel = sequenceLabelsTrace.data[i]

      if sequenceLabel is not None and not resetsTrace.data[i]:
        self._mmData["activeCellsListForSequence"][sequenceLabel].append(
          activeCells)

    for sequenceLabel, activeCellsList in (
        self._mmData["activeCellsListForSequence"].iteritems()):
      confusionMatrix = []

      for i in xrange(len(activeCellsList)):
        row = []

        for j in xrange(len(activeCellsList)):
          row.append(len(activeCellsList[i] ^ activeCellsList[j]))

        confusionMatrix.append(row)

        self._mmData["activeCellsUnionForSequence"][sequenceLabel].update(
          activeCellsList[i])

      self._mmData["stabilityConfusion"][sequenceLabel] = confusionMatrix

    activeCellsUnionForSequenceItems = list(
      self._mmData["activeCellsUnionForSequence"].iteritems())

    for i in xrange(len(activeCellsUnionForSequenceItems)):
      self._mmData["distinctnessConfusionLabels"].append(
        activeCellsUnionForSequenceItems[i][0])
      row = []

      for j in xrange(len(activeCellsUnionForSequenceItems)):
        row.append(len(activeCellsUnionForSequenceItems[i][1] &
                       activeCellsUnionForSequenceItems[j][1]))

      self._mmData["distinctnessConfusionMatrix"].append(row)

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
