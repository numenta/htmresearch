"""
Load, shuffle and plot accelerometer data.
"""

import csv
from htmresearch.frameworks.classification.utils.sensor_data import (
  plotSensorData)



def loadAccelerometerData(dataDir, fileNames):
  """
  Load accelerometer data CSV files and concatenate the data
  :param dataDir: (str) parent file of the input data
  :param fileNames: (list of str) names of the input CSV files
  :return data: (list of list)concatenated accelerometer data
  :return headers: (list of str) headers of the first file that was loaded. All 
    files should have the same headers.
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
            csvWriter.writerow([rowCounter,
                                row[value_idx],
                                int(row[category_idx])])
            rowCounter += 1

  return outputFiles



def main():
  _DATA_DIR = 'data/sensortag'
  _INFILES = ['walk-5min.csv',
              #'run-5min.csv',
              #'stairs-up-5min.csv',
              #'sit-5min.csv',
              'jump-5min.csv',
              #'stairs-down-5min.csv',
              #'stumble-5min.csv'
              ]
  _METRICS = ['x', 'y', 'z']
  _OUTFILE = 'data/sensortag_%s.csv'
  _SLICES = 2

  data, headers, categories = loadAccelerometerData(_DATA_DIR, _INFILES)
  outputFiles = writeData(_OUTFILE, data, headers, _METRICS, categories,
                          _SLICES)

  categoryLabels = [f[:-9] for f in _INFILES]
  for outputFile in outputFiles:
    plotSensorData([outputFile], markers=False, categoryLabels=categoryLabels)



if __name__ == '__main__':
  main()
