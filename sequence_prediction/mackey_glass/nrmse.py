#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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
import csv
import math
import sys

import numpy



def run(filename):
  with open(filename, 'rU') as infile:
    reader = csv.reader(infile)
    reader.next()

    line = reader.next()
    actual = float(line[1])
    predicted = float(line[2])

    actuals = []
    aggregatedError = 0
    n = 0

    for line in reader:
      actual = float(line[1])
      actuals.append(actual)

      aggregatedError += (actual - predicted)**2
      n += 1

      predicted = float(line[2])

    rmse = math.sqrt(aggregatedError / float(n))
    nrmse = rmse / numpy.std(actuals)

    print "RMSE:", rmse
    print "NRMSE:", nrmse



if __name__ == "__main__":
  run(sys.argv[1])
