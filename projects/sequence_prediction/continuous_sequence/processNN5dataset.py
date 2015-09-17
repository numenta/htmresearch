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

import pandas as pd
import csv

def saveSeriesToCSV(fileName, dataSet):
  outputFile = open(fileName,"w")
  csvWriter = csv.writer(outputFile)
  csvWriter.writerow(['date', 'data'])
  csvWriter.writerow(['datetime', 'float'])
  csvWriter.writerow(['T', ''])
  for r in range(len(dataSet)):
    csvWriter.writerow([str(dataSet.index[r]), dataSet[r]])
  outputFile.close()


df = pd.read_excel('./data/NN5dataset.xlsx', header=0, skiprows=[1, 2, 3], index_col=0)

(numRec, numFile) = df.shape

numRecTrain = 735

# break into series and train/test dataset
for i in range(numFile):
  dataSetName = df.columns[i]
  print " save data: ", dataSetName
  dataSet = pd.Series(df[dataSetName])
  trainfileName = './data/NN5/' + dataSetName + '.csv'
  testfileName = './data/NN5/' + dataSetName + '_cont.csv'

  saveSeriesToCSV(trainfileName, dataSet[:numRecTrain])
  saveSeriesToCSV(testfileName, dataSet[numRecTrain:])


