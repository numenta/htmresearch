# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

from baseline_utils import load_traces, save_traces, get_file_name, CONFIG

# Split between train, validation and test
exp_name = 'body_acc_x'
headers = [exp_name, 'sdr', 'label']
datasets_boundaries = {
  'train': [10000, 12000],
  'val': [19300, 21300],
  'test': [26000, 28000]
}

# Load data
min_idx = datasets_boundaries['train'][0]
max_idx = datasets_boundaries['test'][1]

# load traces
file_name = get_file_name(exp_name, CONFIG)
traces = load_traces(file_name, min_idx, max_idx)

# Save data
for phase, idx in datasets_boundaries.items():
  start = idx[0] - min_idx
  end = idx[1] - min_idx
  save_traces('%s_%s' % (phase, exp_name), traces, start, end)
