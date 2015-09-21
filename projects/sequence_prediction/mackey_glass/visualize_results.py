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
import csv
import sys

from matplotlib import pyplot
import numpy



def run(filename, predictionDelay):
  with open(filename, 'rU') as infile:
    reader = csv.reader(infile)
    reader.next()

    actuals = []
    shifted = []

    shifted += [0] * predictionDelay

    for row in reader:
      actuals.append(row[1])
      shifted.append(row[2])

    shifted = shifted[predictionDelay:len(actuals)]
    actuals = actuals[predictionDelay:]
    errors = abs(numpy.array(shifted, dtype=float) - numpy.array(actuals, dtype=float)).tolist()

    pyplot.subplot(2, 1, 1)
    pyplot.plot(shifted)
    pyplot.plot(actuals)

    pyplot.subplot(2, 1, 2)
    pyplot.plot(errors)

    pyplot.show()



if __name__ == "__main__":
  predictionDelay = int(sys.argv[2]) if len(sys.argv) > 2 else 1
  run(sys.argv[1], predictionDelay)
