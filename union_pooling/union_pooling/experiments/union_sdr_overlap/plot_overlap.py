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

import argparse
import csv
import os

import numpy

from experiments.capacity import data_utils



def main(inputPath, outputPath):
  print "Computing Union SDR overlap between SDR traces in following dir:"
  print inputPath + "\n"

  files = os.listdir(inputPath)
  assert len(files) == 2
  pathNoLearn = inputPath + "/" + files[0]
  pathLearn = inputPath + "/" + files[1]

  print "Comparing files..."
  print pathLearn
  print pathNoLearn + "\n"

  # Load source A
  with open(pathLearn, "rU") as fileA:
    csvReader = csv.reader(fileA)
    dataA = [line for line in csvReader]

  # Plot union size for data A
  unionSizeA = [len(datum) for datum in dataA]
  x = [i for i in xrange(len(dataA))]
  stdDevs = None
  data_utils.getErrorbarFigure("Union Size with Learning vs. Time", x,
                               unionSizeA, stdDevs, "Time","Union Size")

  # Load source B
  with open(pathNoLearn, "rU") as fileB:
    csvReader = csv.reader(fileB)
    dataB = [line for line in csvReader]

  # Plot union size for data B
  unionSizeB = [len(datum) for datum in dataB]
  data_utils.getErrorbarFigure("Union Size, no Learning vs. Time", x,
                               unionSizeB, stdDevs, "Time","Union Size")

  assert len(dataA) == len(dataB)

  with open(outputPath + "_overlaps.csv", "wb") as outputFile:
    csvWriter = csv.writer(outputFile)
    overlaps = [getOverlap(dataA[i], dataB[i]) for i in xrange(len(dataA))]
    csvWriter.writerow(overlaps)
    outputFile.flush()

  data_utils.getErrorbarFigure("Overlap vs. Time", x, overlaps, stdDevs,
                               "Time","Overlap")
  raw_input("Press any key to exit...")



def getOverlap(listA, listB):
  arrayA = numpy.array(listA)
  arrayB = numpy.array(listB)
  intersection = numpy.intersect1d(arrayA, arrayB)
  return len(intersection)



def _getArgs():
  """
  Parses and returns command line arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", help="Path to unionSdrTrace .csv files")
  parser.add_argument("--output", help="Path to output csv")
  return parser.parse_args()



if __name__ == "__main__":
  args = _getArgs()
  main(args.input, args.output)
