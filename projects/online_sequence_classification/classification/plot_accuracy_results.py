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

from optparse import OptionParser

from htmresearch.frameworks.classification.utils.traces import (loadTraces,
                                                                plotTraces)

"""
Script to visualize the TM states
Example Usage:

python plot_experiment_result.py --xlim "1700,1750"

"""



def _getArgs():
  parser = OptionParser(usage="Plot trace files of the sequence classification"
                              " experiment ")
  parser.add_option("-f",
                    "--fileName",
                    type=str,
                    dest="fileName",
                    help="File name of the CSV trace file. "
                         "Run 'run_htm_network.py' to generate trace file.")

  parser.add_option("--xlim",
                    type=str,
                    default=None,
                    dest="xl",
                    help="x-axis range")

  parser.add_option("--numTmCells",
                    type=int,
                    default=32 * 2048,
                    dest="numTmCells",
                    help="Number of cells in the Temporal Memory")

  (options, remainder) = parser.parse_args()
  return options, remainder



if __name__ == "__main__":
  (_options, _args) = _getArgs()
  fileName = _options.fileName

  if _options.xl:
    xl = [float(x) for x in _options.xl.split(',')]
  else:
    xl = _options.xl

  traces = loadTraces(fileName)

  numTmCells = _options.numTmCells

  plotTraces(numTmCells, xl, traces)
