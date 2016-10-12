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

INPUT_DIR = os.path.join(parentDir, 'data')
OUTPUT_DIR = os.path.join(parentDir, 'results')
RESULTS_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'seq_classification_results.csv')
FILE_NAMES = [
  'binary_ampl=10.0_mean=0.0_noise=0.0.csv',
  'binary_ampl=10.0_mean=0.0_noise=1.0.csv',
  'sensortag_z.csv'
]
INPUT_FILES = [os.path.join(INPUT_DIR, f) for f in FILE_NAMES]

TRACES_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'traces_%s.csv')

# Verbosity of network
VERBOSITY = 0

HTM_NETWORK_CONFIGS = os.path.join(parentDir, 'htm_network_config',
                                   'network_configs.json')
PLOT_RESULTS = False

# Clustering params
CLUSTERING = True

MERGE_THRESHOLD = 0.4
ANOMALOUS_THRESHOLD = 0.8
STABLE_THRESHOLD = 0.4
MIN_CLUSTER_SIZE = 1
SIMILARITY_THRESHOLD = 0.0
CELLS_TO_CLUSTER = 'tmActiveCells'

# Rolling average calculations
ROLLING_ACCURACY_WINDOW = 10
IGNORE_NOISE = False
ANOMALY_SCORE = 'rollingAnomalyScore'
