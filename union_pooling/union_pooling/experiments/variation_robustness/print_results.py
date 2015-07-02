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

"""Simple utility to print out variation robustness experiment results from a
directory of files.
"""

import csv
from optparse import OptionParser
from os import listdir
from os.path import isfile, join
import sys



SIG_FIGS = 5



def main(outputFilesDir):
  fileList = sorted([f for f in listdir(outputFilesDir)
                     if isfile(join(outputFilesDir, f))])
  for f in fileList:
    with open(join(outputFilesDir, f), "rU") as outputFile:
      csvReader = csv.reader(outputFile)
      csvReader.next()
      print "{1:.{0}f}".format(SIG_FIGS, float(csvReader.next()[0]))



def _getArgs():
  parser = OptionParser(usage="%prog OUTPUT_FILES_DIR "
                              "\n\nSpecify dir containing output files.")
  (options, args) = parser.parse_args(sys.argv[1:])
  if len(args) < 1:
    parser.print_help(sys.stderr)
    sys.exit()

  return args



if __name__ == "__main__":
  _args = _getArgs()
  main(_args[0])
