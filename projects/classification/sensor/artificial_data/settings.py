#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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


# Signal types can be: 'binary', 'sine', 'triangle'
SIGNAL_TYPES = [
  'binary',
  'sine', 
  'triangle'
]

# Parameters to generate the artificial sensor data
DATA_DIR = 'data'
NUM_CATEGORIES = [2]
WHITE_NOISE_AMPLITUDES = [0.0]
SIGNAL_AMPLITUDES = [10.0]
SIGNAL_MEANS = [0.0]
NOISE_LENGTHS = [10]

# Number of phases. Eg: Train (1) SP, (2) TM, (3) TP, (4) Classifier, (5) Test
NUM_PHASES = [5]

# Number of time each phase repeats
NUM_REPS = [5]

# Verbosity of network
VERBOSITY = 0

# Whether to use a JSON config file of the template config generator.
# See network_configs.json for an example of standard config file.
USE_CONFIG_TEMPLATE = False

# Whether or not to plot experiments data and results
PLOT  = False
