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

from nupic.research.monitor_mixin.metric import Metric
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.trace import IndicesTrace, StringsTrace



class TemporalPoolerMonitorMixin(MonitorMixinBase):
  """
  Mixin for TemporalPooler that stores a detailed history, for inspection and
  debugging.
  """

  def getTraceActiveColumns(self):
    """
    @return (Trace) Trace of active columns
    """
    return self._traces["activeColumns"]


  def getTraceSequenceLabels(self):
    """
    @return (Trace) Trace of sequence labels
    """
    return self._traces["sequenceLabels"]


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

    self._traces["activeColumns"].data.append(activeColumns)
    self._traces["sequenceLabels"].data.append(sequenceLabel)


  def getDefaultTraces(self, verbosity=1):
    traces = [
      self.getTraceActiveColumns()
    ]

    if verbosity == 1:
      traces = [trace.makeCountsTrace() for trace in traces]

    return traces + [self.getTraceSequenceLabels()]


  def getDefaultMetrics(self, verbosity=1):
    return ([Metric.createFromTrace(trace)
            for trace in self.getDefaultTraces()[:-1]])


  def clearHistory(self):
    super(TemporalPoolerMonitorMixin, self).clearHistory()

    self._traces["activeColumns"] = IndicesTrace(self, "active columns")
    self._traces["sequenceLabels"] = StringsTrace(self, "sequence labels")
