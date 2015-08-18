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
"""
Utilities to process and visualize data from the sensorimotor experiment
"""

import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
from pylab import rcParams



def combineCsvFiles(directoryPath, outputFileName):
  """
  Combines all csv files in specified path.
  All files are assumed to have a header row followed by data.
  The resulting file contains only 1 head but all of the files' data
  combined.
  Caution, the csv are iterated over in alphabetical order so a file
  100.csv may be appended before a file 10.csv and may mess up yo' data plotting
  """
  appendHeader = True

  # Create csv output writer
  os.chdir(directoryPath)
  with open(outputFileName, "wb") as outputFile:
    csvWriter = csv.writer(outputFile)

    # Iterate over csv files in directory
    for csvFileName in glob.glob("*.csv"):

      # Ignore and write over old version of the same file name
      if csvFileName != outputFileName:

        # Read each file writing the pertinent file lines to output
        with open(csvFileName, "rU") as inputFile:
          csvReader = csv.reader(inputFile)
          line = next(csvReader)
          if appendHeader:
            csvWriter.writerow(line)
            appendHeader = False
          line = next(csvReader)
          csvWriter.writerow(line)



def getChartData(path, xDataColumnIdx, yDataColumnIdxs, yStdDevIdxs):
  """
  Gets chart-ready data from the specified csv file
  """
  assert len(yDataColumnIdxs) == len(yStdDevIdxs)

  with open(path, "rU") as inputFile:
    csvReader = csv.reader(inputFile)

    # Get DV values
    isHeader = True
    xData = []
    for row in csvReader:
      if isHeader:
        isHeader = False
      else:
        xData.append(float(row[xDataColumnIdx]))

    # Get IVs' values
    allYData = []
    allYStdDevs = []
    plotTitles = []

    for i, yColIdx in enumerate(yDataColumnIdxs):
      # Reset the file position to allow iterator reuse
      inputFile.seek(0)

      # build the y data and y std devs
      yCol = []
      yColStdDev = []
      isHeader = True
      stdDevIdx = yStdDevIdxs[i]
      for row in csvReader:
        if isHeader:
          plotTitles.append(row[yColIdx])
        else:
          yCol.append(float(row[yColIdx]))

        # Std Devs
        if isHeader:
          isHeader = False
        elif stdDevIdx == -1:
          yColStdDev.append(0)
        else:
          yColStdDev.append(float(row[stdDevIdx]))

      allYData.append(yCol)
      allYStdDevs.append(yColStdDev)

  return xData, allYData, allYStdDevs, plotTitles



def getErrorbarFigures(title, X, Ys, stdDevs, plotTitles, xAxisLabel,
                      yAxisLabels, gridFormat):
  """
  Plots the specified data and saves specified plot to file
  """
  rcParams['figure.figsize'] = 15, 15
  fig = plt.figure()
  fig.suptitle(title)
  fig.subplots_adjust(left=None, right=None, bottom=None, top=None,
                      wspace=None, hspace=0.35)
  plt.ion()
  plt.show()
  rcParams.update({'font.size': 12})

  for i, y in enumerate(Ys):
    ax = fig.add_subplot(gridFormat + 1 + i)
    ax.set_title(plotTitles[i])
    ax.set_xlabel(xAxisLabel)
    ax.set_ylabel(yAxisLabels[i])
    ax.axis([0, max(X) + 10, 0, 20])
    ax.errorbar(X, y, stdDevs[i])

  return fig



def getErrorbarFigure(title, x, y, stdDevs, xAxisLabel, yAxisLabel,
                      xRangeMax=None, yRangeMax=None):
  fig = plt.figure()
  fig.suptitle(title)
  fig.subplots_adjust(left=None, right=None, bottom=None, top=None,
                      wspace=None, hspace=0.35)
  plt.ion()
  plt.show()

  ax = fig.add_subplot(111)
  ax.set_xlabel(xAxisLabel)
  ax.set_ylabel(yAxisLabel)

  if xRangeMax is None:
    xRangeMax = max(x) + 10
  if yRangeMax is None:
    yRangeMax = max(y) + 10

  ax.axis([0, xRangeMax, 0, yRangeMax])
  ax.errorbar(x, y, stdDevs)

  plt.draw()
  return fig


def plotSensorimotorExperimentResults(filesDir, outputFileName):
  """
  Plots the data produced by
  sensorimotor/experiments/capacity/run.py
  """
  print "Combining csv's in: {0}".format(filesDir)
  print "Output file name: {0}\n".format(outputFileName)
  combineCsvFiles(filesDir, outputFileName + ".csv")

  # 0 when number of worlds is IV
  # 1 when number of elements is IV
  xColumnIdx = 0
  xAxisLabel = "Worlds"
  yAxisLabels = ["Cells", "Cells", "Cells", "Cells", "Cols", "Cols"]

  # Following indices are columns in the excel file produced by
  # sensorimotor/experiments/capacity/run.py and represent the following
  # metrics:
  # Mean & Max Stability, Mean & Max Distinctness, Mean & Max Bursting Cols
  yColumnIdxs = [11, 9, 16, 14, 46, 44]

  # The following are the column indices in the same xls file for the std
  # deviations of the metrics specified by yColumnIdxs. A -1 means the script
  # won't plot a std dev for the corresponding metric.
  yStdDevIdxs = [12, -1, 17, -1, 47, -1]
  iv, dvs, stdDevs, metricTitles = getChartData(outputFileName + ".csv",
                                                xColumnIdx, yColumnIdxs,
                                                yStdDevIdxs)

  # 3x2 subplot grid
  gridFormat = 320
  getErrorbarFigures(filesDir, iv, dvs, stdDevs, metricTitles,
                     xAxisLabel, yAxisLabels, gridFormat)
  plt.savefig(outputFileName, bbox_inches="tight")
  plt.draw()
  raw_input("Press enter...")



if __name__ == "__main__":
  if len(sys.argv) < 3:
    print "Usage: ./data_utils.py FILES_DIR COMBINED_FILE_NAME"
    sys.exit()
  plotSensorimotorExperimentResults(sys.argv[1], sys.argv[2])
