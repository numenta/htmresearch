#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
This program runs many experiments in parallel.
"""

import importlib
from multiprocessing import Pool

from optparse import OptionParser
import os
from os import listdir
from os.path import isfile, join
import sys
import traceback
import yaml



def _getArgs():
  parser = OptionParser(usage="%prog EXPERIMENT_MODULE FUNCTION_MODULE "
                              "PARAM_FILES_DIR OUTPUT_DIR [options]"
                              "\n\nFrom MODULE runs FUNCTION for each "
                              "parameter file in PARAM_FILES_DIR in parallel "
                              "outputting to OUTPUT_DIR.")
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

  (options, args) = parser.parse_args(sys.argv[1:])
  if len(args) != 4:
    parser.print_help(sys.stderr)
    sys.exit()

  return options, args



def run(args):
  (function, params, paramsDir, paramFileName, paramFile,
  outputDir, consoleVerbosity, plotVerbosity) = args
  logPath = "{0}.log".format(paramFileName)

  with open(logPath, "w", buffering=0) as logFile:
    sys.stdout = logFile
    exception = None

    try:
      function(params, paramsDir, outputDir, consoleVerbosity, plotVerbosity)
    except Exception, err:
      print traceback.format_exc()
      exception = err

    sys.stdout = sys.__stdout__
    if exception is None:
      print "Finished ({0}).".format(logPath)
    else:
      print "Failed with exception: {0} ({1}).".format(exception, logPath)



def loadExperiments():
  (options, args) = _getArgs()

  moduleName = args[0]
  functionName = args[1]
  paramsDir = args[2]
  outputDir = args[3]
  workers = options.workers
  plotVerbosity = options.plotVerbosity
  consoleVerbosity = options.consoleVerbosity

  experimentModule = importlib.import_module(moduleName)
  experimentFunction = getattr(experimentModule, functionName)

  print "Param files dir: {0}".format(paramsDir)
  print "Output dir: {0}\n".format(outputDir)

  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

  pool = Pool(processes=workers)
  paramFileList = [f for f in listdir(paramsDir) if isfile(join(paramsDir,f)) ]
  for f in paramFileList:
    with open(join(paramsDir,f)) as paramFile:
      params = yaml.safe_load(paramFile)
      print "Starting {0} ...".format(f)
      runArgs = [(experimentFunction, params, paramsDir, f, paramFile,
                  outputDir, consoleVerbosity, plotVerbosity)]
      pool.map(run, runArgs)


if __name__ == "__main__":
  loadExperiments()
