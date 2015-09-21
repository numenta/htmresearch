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
import numpy as np

angles = np.linspace(0, 120, num=2400)
sine = np.sin(angles*4)

outputFile = open('sine.csv',"w")
csvWriter = csv.writer(outputFile)
csvWriter.writerow(['angle', 'data'])
csvWriter.writerow(['float', 'float'])
csvWriter.writerow(['', ''])
for i in range(2000):
  csvWriter.writerow([angles[i], sine[i]])
outputFile.close()


outputFile = open('sine_cont.csv',"w")
csvWriter = csv.writer(outputFile)
csvWriter.writerow(['angle', 'data'])
csvWriter.writerow(['float', 'float'])
csvWriter.writerow(['', ''])
for i in range(2001, 2200):
  csvWriter.writerow([angles[i], sine[i]])
outputFile.close()



# outputFile = open('data/sine_der.csv',"w")
# csvWriter = csv.writer(outputFile)
# csvWriter.writerow(['angle', 'data'])
# csvWriter.writerow(['float', 'float'])
# csvWriter.writerow(['', ''])
# for i in range(1,2000):
#   csvWriter.writerow([angles[i], sine[i]-sine[i-1]])
# outputFile.close()
#
#
# outputFile = open('data/sine_der_cont.csv',"w")
# csvWriter = csv.writer(outputFile)
# csvWriter.writerow(['angle', 'data'])
# csvWriter.writerow(['float', 'float'])
# csvWriter.writerow(['', ''])
# for i in range(2001, 2200):
#   csvWriter.writerow([angles[i], sine[i]-sine[i-1]])
# outputFile.close()
