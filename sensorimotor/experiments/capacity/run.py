#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
"""
This program runs many capacity experiments in parallel.
"""
from multiprocessing import Pool
from optparse import OptionParser
import os
import sys
import traceback
import yaml

from experiments.capacity import experiment



DEFAULT_OUTPUT_DIR = "output"



parser = OptionParser(
  usage="%prog path/to/defs.yaml path/to/params.yaml [options]"
        "\n\nRun experiments in defs.py using parameters from params.py "
        "in parallel."
)
parser.add_option(
  "-o",
  "--output-dir",
  default=DEFAULT_OUTPUT_DIR,
  dest="outputDir",
  help="Output directory to write the results and logs to."
)
parser.add_option(
  "-w",
  "--workers",
  type=int,
  default=4,
  dest="workers",
  help="Max number of parallel workers."
)



def consolePrint(string):
  print string


def run(args):
  defn, params, outputDir = args
  numWorlds = defn["worlds"]
  numElements = defn["elements"]

  filePrefix = os.path.join(outputDir, "{0}x{1}".format(numWorlds,
                                                        numElements))
  logPath = "{0}.log".format(filePrefix)

  print "Starting ({0})...".format(logPath)

  with open(logPath, "w", buffering=0) as logFile:
    sys.stdout = logFile
    exception = None

    try:
      experiment.run(numWorlds, numElements, outputDir, params=params)
    except Exception, err:
      print traceback.format_exc()
      exception = err

    sys.stdout = sys.__stdout__
    if exception is None:
      print "Finished ({0}).".format(logPath)
    else:
      print "Failed with exception: {0} ({1}).".format(exception, logPath)



if __name__ == "__main__":
  (options, args) = parser.parse_args(sys.argv[1:])
  if len(args) < 2:
    parser.print_help(sys.stderr)
    sys.exit()

  defsPath = args[0]
  paramsPath = args[1]
  outputDir = options.outputDir
  workers = options.workers

  with open(defsPath) as defsFile:
    defs = yaml.safe_load(defsFile)

    with open(paramsPath) as paramsFile:
      params = yaml.safe_load(paramsFile)

      if not os.path.exists(outputDir):
        os.makedirs(outputDir)

      pool = Pool(processes=workers)
      pool.map(run, [(defn, params, outputDir) for defn in defs])

