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

"""
Load, shuffle and plot accelerometer data.
"""

import csv
import json
import os

from htmresearch.frameworks.classification.utils.sensor_data import (
  plotSensorData)
from settings.acc_data import (DATA_DIR,
                               INPUT_FILES,
                               OUTPUT_DIR,
                               METRICS,
                               SLICES,
                               MAX_POINTS)



def loadAccelerometerData(dataDir, fileNames):
  """
  Load accelerometer data CSV files and concatenate the data
  :param dataDir: (str) parent file of the input data
  :param fileNames: (list of str) names of the input CSV files
  :return data: (list of list)concatenated accelerometer data
  :return headers: (list of str) headers of the first file that was loaded. 
    All files should have the same headers.
  :return categories: (dict) start and end indices of each category
  """
  data = []
  categories = []
  totalRowCounter = 0
  headers = None
  for fileName in fileNames:
    inFile = '%s/%s' % (dataDir, fileName)
    category = fileNames.index(fileName) + 1
    rowCounter = 0
    with open(inFile, 'rb') as f:
      csvReader = csv.reader(f)
      h = csvReader.next() + ['category']
      if not headers:
        headers = h
      else:
        assert headers == h
      for row in csvReader:
        row.append(category)
        data.append(row)
        rowCounter += 1

    categories.append({
      'category': category,
      'start': totalRowCounter,
      'end': totalRowCounter + rowCounter
    })
    totalRowCounter += rowCounter
  return data, headers, categories



def writeData(outFileBaseName, inputData, inputHeaders, metrics,
              categories, slices):
  """
  Write data to CSV
  :param outFile: (str) base file name of the output CSV files
  :param inputData: (list of list) input data that will be written to file
  :param inputHeaders: (list of str) headers of the input data
  :param metrics: (list of str) sub-list of the input headers used to 
    generate the output files (one file per metric)
  :param categories: (dict) start and end indices for each category 
  :param slices: (int) number of times the data should be sliced per category. 
  The sliced up data will be shuffled in order to avoid big chunks of data 
  with the same category.
  """
  outputFiles = []
  for metric in metrics:
    outFile = outFileBaseName % metric
    outputFiles.append(outFile)
    if metric in inputHeaders:
      value_idx = inputHeaders.index(metric)
    if 'category' in inputHeaders:
      category_idx = inputHeaders.index('category')
    rowCounter = 0
    with open(outFile, 'wb') as f:
      csvWriter = csv.writer(f)
      csvWriter.writerow(['x', 'y', 'label'])
      csvWriter.writerow(['float', 'float', 'int'])
      csvWriter.writerow([None, None, 'C'])
      for i in range(slices):
        for categoryInfo in categories:
          step = (categoryInfo['end'] - categoryInfo['start']) / slices
          start = categoryInfo['start'] + i * step
          end = categoryInfo['start'] + (i + 1) * step
          for row in inputData[start:end]:
            if rowCounter > MAX_POINTS:
              break
            csvWriter.writerow([rowCounter,
                                row[value_idx],
                                int(row[category_idx])])
            rowCounter += 1

  return outputFiles



def main():
  data, headers, categories = loadAccelerometerData(DATA_DIR, INPUT_FILES)
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  outputFileTemplate = os.path.join(OUTPUT_DIR, 'sensortag_%s.csv')
  outputFiles = writeData(outputFileTemplate, data, headers, METRICS,
                          categories, SLICES)

  categoryLabels = [f[:-9] for f in INPUT_FILES]
  for outputFile in outputFiles:
    plotSensorData([outputFile], markers=False, categoryLabels=categoryLabels)



if __name__ == '__main__':
  main()
