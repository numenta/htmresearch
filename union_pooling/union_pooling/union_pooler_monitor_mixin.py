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

"""
Union Pooler mixin that enables detailed monitoring of history.
"""

import numpy

from nupic.research.monitor_mixin.metric import Metric
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.plot import Plot
from nupic.research.monitor_mixin.trace import (IndicesTrace, StringsTrace,
                                                BoolsTrace, MetricsTrace)

class UnionPoolerMonitorMixin(MonitorMixinBase):


  def __init__(self, *args, **kwargs):
    super(UnionPoolerMonitorMixin, self).__init__(*args, **kwargs)

    self._mmResetActive = True  # First iteration is always a reset
