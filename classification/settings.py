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

# start and end indices of each class (inclusive)
CLASS_RANGES = {'label0': [{'start': 0, 'end': 499}, {'start': 1000, 'end': 1499}, {'start': 2000, 'end': 2499}, {'start': 3000, 'end': 3499}],
                'label1': [{'start': 500, 'end': 749}, {'start': 1500, 'end': 1749}, {'start': 2500, 'end': 2749}, {'start': 3500, 'end': 3749}],
                'label2': [{'start': 750, 'end': 999}, {'start': 1750, 'end': 1999}, {'start': 2750, 'end': 2999}, {'start': 3750, 'end': 3999}]
                }

SIGNAL_TYPES = ["no_noise", "white_noise"]
RESULTS_DIR = "results"
DATA_DIR = "data" 
MODEL_PARAMS_DIR = 'model_params'
TRAINING_SET_SIZE=3000
WHITE_NOISE_AMPLITUDE = 0.5
SIGNAL_AMPLITUDE = 1
SIGNAL_MEAN = 10
