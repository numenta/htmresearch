#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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


SIGNAL_TYPES = ["no_noise", "white_noise"]
RESULTS_DIR = "results"
DATA_DIR = "data" 
MODEL_PARAMS_DIR = 'model_params'
SEQUENCE_LENGTH = 200
NUMBER_OF_LABELS = 3 
NUM_RECORDS = 2400
SP_TRAINING_SET_SIZE = NUM_RECORDS * 1/4  
TM_TRAINING_SET_SIZE = NUM_RECORDS * 1/2  
CLASSIFIER_TRAINING_SET_SIZE = NUM_RECORDS * 3/4 
DEFAULT_WHITE_NOISE_AMPLITUDE = 10.0
WHITE_NOISE_AMPLITUDE_RANGES = [0, 1, 10, 100]
SIGNAL_AMPLITUDE = 1.0
SIGNAL_MEAN = 1.0
SIGNAL_PERIOD = 20.0
