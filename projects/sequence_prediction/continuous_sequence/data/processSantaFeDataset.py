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

def convertDatToCSV(inputFileName, outputFileName, Nrpts=1, maxLength=None):
  df = pd.read_table(inputFileName, header=None, names=['value'])
  if maxLength is None:
    maxLength = len(df)
  outputFile = open(outputFileName,"w")
  csvWriter = csv.writer(outputFile)
  if Nrpts==1:
    csvWriter.writerow(['step', 'data'])
    csvWriter.writerow(['float', 'float'])
    csvWriter.writerow(['', ''])
    for i in xrange(maxLength):
      csvWriter.writerow([i,df['value'][i]])

  else:
    csvWriter.writerow(['step', 'data', 'reset'])
    csvWriter.writerow(['float', 'float', 'int'])
    csvWriter.writerow(['', '', 'R'])

    for _ in xrange(Nrpts):
      for i in xrange(maxLength):
        if i==0:
          reset = 1
        else:
          reset = 0
        csvWriter.writerow([i, df['value'][i], reset])

  outputFile.close()


inputFileName = 'SantaFe_A_cont.dat'
outputFileName = 'SantaFe_A_cont.csv'
convertDatToCSV(inputFileName, outputFileName, maxLength=100)

inputFileName = 'SantaFe_A.dat'
outputFileName = 'SantaFe_A.csv'
convertDatToCSV(inputFileName, outputFileName, Nrpts=1)
