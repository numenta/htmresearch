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
from nupic.research.monitor_mixin.trace import (
  IndicesTrace, StringsTrace,  BoolsTrace)



class TemporalPoolerMonitorMixin(MonitorMixinBase):
  """
  Mixin for TemporalPooler that stores a detailed history, for inspection and
  debugging.
  """

  def __init__(self, *args, **kwargs):
    super(TemporalPoolerMonitorMixin, self).__init__(*args, **kwargs)

    self._resetActive = True  # First iteration is always a reset


  def getTraceActiveCells(self):
    """
    @return (Trace) Trace of active cells
    """
    return self._traces["activeCells"]


  def getTraceSequenceLabels(self):
    """
    @return (Trace) Trace of sequence labels
    """
    return self._traces["sequenceLabels"]


  def getTraceResets(self):
    """
    @return (Trace) Trace of resets
    """
    return self._traces["resets"]


  def getDataStabilityConfusion(self):
    self._computeSequenceRepresentationData()
    return self._data["stabilityConfusion"]


  def getMetricStabilityConfusion(self):
    data = self.getDataStabilityConfusion()
    numberLists = [item for sublist in data.values() for item in sublist]
    numbers = [item for sublist in numberLists for item in sublist]
    return Metric(self, "stability confusion", numbers)


  def prettyPrintDataStabilityConfusion(self):
    data = self.getDataStabilityConfusion()
    text = ""

    for sequenceLabel, confusionMatrix in data.iteritems():
      text += "Confusion matrix for {0}:\n\n".format(sequenceLabel)
      text += ("\n".join([''.join(['{:4}'.format(item) for item in row])
                          for row in confusionMatrix]))
      text += "\n\n"

    return text


  # ==============================
  # Helpers
  # ==============================

  def _computeSequenceRepresentationData(self):
    if not self._sequenceRepresentationDataStale:
      return

    self._data["activeCellsListForSequence"] = defaultdict(list)
    self._data["stabilityConfusion"] = {}

    activeCellsTrace = self.getTraceActiveCells()
    sequenceLabelsTrace = self.getTraceSequenceLabels()
    resetsTrace = self.getTraceResets()

    for i, activeCells in enumerate(activeCellsTrace.data):
      sequenceLabel = sequenceLabelsTrace.data[i]

      if sequenceLabel is not None and not resetsTrace.data[i]:
        self._data["activeCellsListForSequence"][sequenceLabel].append(
          activeCells)

    for sequenceLabel, activeCellsList in (
        self._data["activeCellsListForSequence"].iteritems()):
      confusionMatrix = []

      for i in xrange(len(activeCellsList)):
        row = []

        for j in xrange(len(activeCellsList)):
          row.append(len(activeCellsList[i] ^ activeCellsList[j]))

        confusionMatrix.append(row)

      self._data["stabilityConfusion"][sequenceLabel] = confusionMatrix

    self._transitionTracesStale = False


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

    self._traces["activeCells"].data.append(activeCells)
    self._traces["sequenceLabels"].data.append(sequenceLabel)

    self._traces["resets"].data.append(self._resetActive)
    self._resetActive = False

    self._sequenceRepresentationDataStale = True


  def reset(self):
    super(TemporalPoolerMonitorMixin, self).reset()

    self._resetActive = True


  def getDefaultTraces(self, verbosity=1):
    traces = [
      self.getTraceActiveCells()
    ]

    if verbosity == 1:
      traces = [trace.makeCountsTrace() for trace in traces]

    return traces + [self.getTraceSequenceLabels()]


  def getDefaultMetrics(self, verbosity=1):
    metrics = ([Metric.createFromTrace(trace)
                for trace in self.getDefaultTraces()[:-1]])
    metrics.append(self.getMetricStabilityConfusion())
    return metrics


  def clearHistory(self):
    super(TemporalPoolerMonitorMixin, self).clearHistory()

    self._traces["activeCells"] = IndicesTrace(self, "active cells")
    self._traces["sequenceLabels"] = StringsTrace(self, "sequence labels")
    self._traces["resets"] = BoolsTrace(self, "resets")

    self._sequenceRepresentationDataStale = True
