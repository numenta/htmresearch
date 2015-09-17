# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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
import numpy as np
import csv

dataSet = 'nyc_taxi'
filePath = dataSet+'.csv'
df = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data', 'timeofday', 'dayofweek'])

# create a new daily profile
dailyTime = np.sort(df['timeofday'].unique())
dailyHour = dailyTime/60
profile = np.ones((len(dailyTime),))
# decrease 7am-11am traffic by 20%
profile[np.where(np.all([dailyHour >= 7.0, dailyHour < 11.0], axis=0))[0]] = 0.8
# increase 21:00 - 24:00 traffic by 20%
profile[np.where(np.all([dailyHour >= 21.0, dailyHour <= 23.0], axis=0))[0]] = 1.2
dailyProfile = {}
for i in range(len(dailyTime)):
  dailyProfile[dailyTime[i]] = profile[i]

# apply the new daily pattern to weekday traffic
old_data = df['data']
new_data = np.zeros(old_data.shape)
for i in xrange(len(old_data)):
  if df['dayofweek'][i] < 5:
    new_data[i] = old_data[i] * dailyProfile[df['timeofday'][i]]
  else:
    new_data[i] = old_data[i]

df['data'] = new_data


# save perturbed data
outputFile = open('nyc_taxi_perturb.csv', "w")
csvWriter = csv.writer(outputFile)
csvWriter.writerow(['timestamp', 'passenger_count', 'timeofday', 'dayofweek'])
csvWriter.writerow(['datetime', 'int', 'int', 'string'])
csvWriter.writerow(['T', '', '', ''])
for i in range(len(df)):
  csvWriter.writerow([df.index[i], df.data[i], df.timeofday[i], df.dayofweek[i]])
outputFile.close()