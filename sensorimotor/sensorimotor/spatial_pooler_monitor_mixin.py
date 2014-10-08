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
Spatial Pooler mixin that enables detailed monitoring of history.
"""

from collections import defaultdict
import numpy
from nupic.research.monitor_mixin.metric import Metric
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.trace import (
  IndicesTrace, CountsTrace, StringsTrace,  BoolsTrace)


class SpatialPoolerMonitorMixin(MonitorMixinBase):
  """
  Mixin for SpatialPooler that stores a detailed history, for inspection and
  debugging.
  """

  def __init__(self, *args, **kwargs):
    super(SpatialPoolerMonitorMixin, self).__init__(*args, **kwargs)


  def mmGetTraceActiveColumns(self):
    """
    @return (Trace) Trace of active columns
    """
    return self._mmTraces["activeColumns"]


  def mmGetTraceNumConnections(self):
    """
    @return (Trace) Trace of # connections
    """
    return self._mmTraces["connections"]


  def mmGetDataDutyCycles(self):
    """
    Computes the duty cycle for all columns
    """
    dutyCycle = [0] * self._numColumns
    activeColumns = self.mmGetTraceActiveColumns().data
    for i in range(len(activeColumns)):
        for j in range(len(activeColumns[i])):
            dutyCycle[activeColumns[i][j]] += 1

    return dutyCycle

  def mmGetMetricDutyCycles(self):
    """
    Metric for duty cycle for all columns
    :return: (Metric) metric
    """
    dutyCycle = self.mmGetDataDutyCycles()

    return Metric(self,
                  "active duty cycle for all columns",
                  dutyCycle)


  # ==============================
  # Overrides
  # ==============================

  def compute(self, inputVector, learn, activeArray, *args, **kwargs):
    super(SpatialPoolerMonitorMixin, self).compute(
        inputVector, learn, activeArray, *args,**kwargs)

    # extract indices of active columns
    activeColumns = numpy.nonzero(activeArray)

    # total number of connections
    connectedCounts = numpy.zeros(self._numColumns, dtype='uint32')
    self.getConnectedCounts(connectedCounts)
    totalConnection = numpy.sum(connectedCounts)

    self._mmTraces["activeColumns"].data.append(activeColumns)
    self._mmTraces["connections"].data.append(totalConnection)


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
    self._mmTraces["connections"] = CountsTrace(self, "connections")