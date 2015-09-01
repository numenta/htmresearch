#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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

parser = OptionParser(usage="%prog path/to/defs.yaml path/to/params.yaml "
                            "[options]"
                            "\n\nRun experiments in defs.py using parameters "
                            "from params.py in parallel.")
parser.add_option("-o",
                  "--output-dir",
                  default=DEFAULT_OUTPUT_DIR,
                  dest="outputDir",
                  help="Output directory to write the results and logs to.")
parser.add_option("-w",
                  "--workers",
                  type=int,
                  default=4,
                  dest="workers",
                  help="Max number of parallel workers.")
parser.add_option("-p",
                  "--plot",
                  type=int,
                  default=0,
                  dest="plotVerbosity",
                  help="Plot verbosity")
parser.add_option("-c",
                  "--console",
                  type=int,
                  default=0,
                  dest="consoleVerbosity",
                  help="Console verbosity")



def run(args):
  defn, params, outputDir, plotVerbosity, consoleVerbosity = args
  numWorlds = defn["worlds"]
  numElements = defn["elements"]

  print "Starting {0} worlds x {1} elems...".format(numWorlds, numElements)
  fileName = "{0:0>3}x{1:0>3}".format(numWorlds, numElements)
  filePrefix = os.path.join(outputDir, fileName)
  logPath = "{0}.log".format(filePrefix)

  with open(logPath, "w", buffering=0) as logFile:
    sys.stdout = logFile
    exception = None

    try:
      experiment.run(numWorlds, numElements, params, outputDir, plotVerbosity,
                     consoleVerbosity)
    except Exception, err:
      print traceback.format_exc()
      exception = err

    sys.stdout = sys.__stdout__
    if exception is None:
      print "Finished ({0}).".format(logPath)
    else:
      print "Failed with exception: {0} ({1}).".format(exception, logPath)



def loadExperiments():
  (options, args) = parser.parse_args(sys.argv[1:])
  if len(args) < 2:
    parser.print_help(sys.stderr)
    sys.exit()

  defsPath = args[0]
  paramsPath = args[1]
  outputDir = options.outputDir
  workers = options.workers
  plotVerbosity = options.plotVerbosity
  consoleVerbosity = options.consoleVerbosity

  print "Defs path: {0}".format(defsPath)
  print "Params path: {0}".format(paramsPath)
  print "Output dir: {0}\n".format(outputDir)
  with open(defsPath) as defsFile:
    defs = yaml.safe_load(defsFile)

    with open(paramsPath) as paramsFile:
      params = yaml.safe_load(paramsFile)

      if not os.path.exists(outputDir):
        os.makedirs(outputDir)

      pool = Pool(processes=workers)
      pool.map(run, [(defn, params, outputDir, plotVerbosity, consoleVerbosity)
                     for defn in defs])



if __name__ == "__main__":
  loadExperiments()
