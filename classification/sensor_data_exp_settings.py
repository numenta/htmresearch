#!/usr/bin/env python
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


# Parameters to generate the artificial sensor data
OUTFILE_NAME = "white_noise"
SEQUENCE_LENGTH = 200
NUM_CATEGORIES = 3
NUM_RECORDS = 2400
WHITE_NOISE_AMPLITUDES = [0.0, 1.0]
SIGNAL_AMPLITUDES = [1.0]
SIGNAL_MEANS = [1.0]
SIGNAL_PERIODS = [20.0]

# Additional parameters to run the classification experiments 
RESULTS_DIR = "results"
MODEL_PARAMS_DIR = 'model_params'
DATA_DIR = "data"

# Verbosity of the debug messages in the classification network
DEBUG_VERBOSITY = 0

# Name of the partition used for the test set
TEST_PARTITION_NAME = "test"


