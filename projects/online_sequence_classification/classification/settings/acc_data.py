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

DATA_DIR = os.path.join(parentDir, 'data', 'sensortag')
INPUT_FILES = [
  'walk-5min.csv',
  'run-5min.csv',
  # 'stairs-up-5min.csv',
  #'sit-5min.csv',
  #'jump-5min.csv',
  # 'stairs-down-5min.csv',
  # 'stumble-5min.csv'
]
METRICS = ['x', 'y', 'z']
OUTPUT_FILE = os.path.join(parentDir, 'data', 'sensortag_%s.csv')
SLICES = 40
MAX_POINTS = 1000

