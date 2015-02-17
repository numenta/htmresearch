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
Combines all output files in a directory into a single file
"""

import csv
import glob
import os



def combineCsvFiles(directoryPath, fileExtension):
  appendHeader = True

  # Create csv output writer
  with open(directoryPath + "/allCombined.csv", "wb") as outputFile:
    csvWriter = csv.writer(outputFile)

    # Iterate over csv files in directory
    os.chdir(directoryPath)
    for filePath in glob.glob("*." + fileExtension):

      # Write the pertinent file lines
      with open(filePath, "rb") as inputFile:
        csvReader = csv.reader(inputFile)
        line = next(csvReader)
        if appendHeader:
          csvWriter.writerow(line)
          appendHeader = False
        line = next(csvReader)
        csvWriter.writerow(line)



if __name__ == "__main__":
  path = "output/strict-varyElements/slow2_13/slow10xRedo"
  combineCsvFiles(path, "csv")
