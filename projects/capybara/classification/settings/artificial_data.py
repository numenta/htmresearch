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
import os

parentDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

DATA_DIR = os.path.join(parentDir, 'data', 'artificial')

# Signal types can be: 'binary', 'sine', 'triangle'
SIGNAL_TYPES = ['binary']

# Parameters to generate the artificial sensor data
NUM_CATEGORIES = [2]
WHITE_NOISE_AMPLITUDES = [0.0, 1.0]
SIGNAL_AMPLITUDES = [10.0]
SIGNAL_MEANS = [0.0]
NOISE_LENGTHS = [10]

# Number of phases.
NUM_PHASES = [1]

# Number of time each phase repeats
# Best: NUM_REPS = [50]
NUM_REPS = [50]

