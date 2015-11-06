# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
Run param finder. This script will be frozen with cx_freeze. 
"""

from htmresearch.frameworks.utils.param_finder import read_csv_files
from htmresearch.frameworks.utils.param_finder \
  import get_suggested_timescale_and_encoder

(timestamps, values) = read_csv_files('example_data/art_daily_flatmiddle.csv')
(new_sampling_interval, useTimeOfDay,
 useDayOfWeek) = get_suggested_timescale_and_encoder(timestamps, values)
