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
import sys

import matplotlib.pyplot as plt
import numpy

from experiments.capacity import data_utils



_OVERLAPS_FILE_NAME = "/overlaps.csv"



def main(inputPath, csvOutputPath, imgOutputPath):
  # remove existing /overlaps.csv if present
  if os.path.exists(csvOutputPath + _OVERLAPS_FILE_NAME):
    os.remove(csvOutputPath + _OVERLAPS_FILE_NAME)

  if not os.path.exists(csvOutputPath):
    os.makedirs(csvOutputPath)

  if not os.path.exists(imgOutputPath):
    os.makedirs(imgOutputPath)

  print "Computing Union SDR overlap between SDR traces in following dir:"
  print inputPath + "\n"

  files = os.listdir(inputPath)
  if len(files) != 2:
    print "Found {0} files at input path. Requires exactly 2.".format(inputPath)
    sys.exit(1)

  pathNoLearn = inputPath + "/" + files[0]
  pathLearn = inputPath + "/" + files[1]

  print "Comparing files..."
  print pathLearn
  print pathNoLearn + "\n"

  # Load source A
  with open(pathLearn, "rU") as fileA:
    csvReader = csv.reader(fileA)
    dataA = [line for line in csvReader]
  unionSizeA = [len(datum) for datum in dataA]

  # Load source B
  with open(pathNoLearn, "rU") as fileB:
    csvReader = csv.reader(fileB)
    dataB = [line for line in csvReader]
  unionSizeB = [len(datum) for datum in dataB]

  assert len(dataA) == len(dataB)

  # To display all plots on the same y scale
  yRangeMax = 1.05 * max(max(unionSizeA), max(unionSizeB))

  # Plot union size for data A
  x = [i for i in xrange(len(dataA))]
  stdDevs = None
  title = "Union Size with Learning vs. Time"
  data_utils.getErrorbarFigure(title, x, unionSizeA, stdDevs, "Time",
                               "Union Size", yRangeMax=yRangeMax)
  figPath = "{0}/{1}.png".format(imgOutputPath, title)
  plt.savefig(figPath, bbox_inches="tight")

  # Plot union size for data B and save image
  title = "Union Size without Learning vs. Time"
  data_utils.getErrorbarFigure(title, x, unionSizeB, stdDevs, "Time",
                               "Union Size", yRangeMax=yRangeMax)
  figPath = "{0}/{1}.png".format(imgOutputPath, title)
  plt.savefig(figPath, bbox_inches="tight")

  with open(csvOutputPath + _OVERLAPS_FILE_NAME, "wb") as outputFile:
    csvWriter = csv.writer(outputFile)
    overlaps = [getOverlap(dataA[i], dataB[i]) for i in xrange(len(dataA))]
    csvWriter.writerow(overlaps)
    outputFile.flush()

  # Plot overlap and save image
  title = "Learn-NoLearn Union SDR Overlap vs. Time"
  data_utils.getErrorbarFigure(title, x, overlaps, stdDevs, "Time","Overlap",
                               yRangeMax=yRangeMax)
  figPath = "{0}/{1}.png".format(imgOutputPath, title)
  plt.savefig(figPath, bbox_inches="tight")

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
  parser.add_argument("--csvOutput", help="Path for csv output.")
  parser.add_argument("--imgOutput", help="Path for image output.")
  return parser.parse_args()



if __name__ == "__main__":
  args = _getArgs()
  main(args.input, args.csvOutput, args.imgOutput)
