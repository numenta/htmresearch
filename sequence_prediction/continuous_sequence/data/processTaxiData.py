import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from pygeocoder import Geocoder

plt.ion()

year = 2015
record_num = []

aggregation_rule = {'Sum': sum}

ts_all = None

for year in [2014, 2015]:
  for month in xrange(1, 13):
    datafileName =  'yellow_tripdata_' + str(year) + '-' + "{:0>2d}".format(month) + '.csv'

    if os.path.isfile(datafileName):
      print " Load Datafile: ", datafileName
      # df = pd.read_csv(datafileName, header=0, nrows=100,  usecols=[1, 3, 5, 6],
      #                  names=['pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude'])
      #
      # postcode = np.zeros(len(df))
      # for i in xrange(len(df)):
      #   try:
      #     results = Geocoder.reverse_geocode(df['pickup_latitude'][i], df['pickup_longitude'][i])
      #     postcode[i] = results.postal_code
      #   except:
      #     pass


      df = pd.read_csv(datafileName, header=0, usecols=[1, 3], names=['pickup_datetime', 'passenger_count'])

      record_num.append(len(df))

      ts = pd.Series(np.array(df.passenger_count), index=pd.to_datetime(df.pickup_datetime))
      del df

      print " aggregate data at 30min resolution"
      ts_aggregate = ts.resample("30min", how=aggregation_rule)

      if ts_all is not None:
        print " concat ts_all"
        ts_all = pd.concat([ts_all, ts_aggregate])
      else:
        print " initialize ts_all"
        ts_all = ts_aggregate
    else:
      print datafileName, " not exist"


print "include time of day and day of week as input field"
date = ts_all.index
dayofweek = (date.dayofweek)
timeofday = (date.hour*60 + date.minute)
passenger_count = np.array(ts_all['Sum'])
seq = pd.DataFrame(np.transpose(np.array([passenger_count, timeofday, dayofweek])), columns=['passenger_count', 'timeofday', 'dayofweek'], index=ts_all.index)

plt.close('all')
plt.figure(1)
plt.plot(seq.index, seq.passenger_count)

import csv
outputFile = open('nyc_taxi.csv',"w")
csvWriter = csv.writer(outputFile)
csvWriter.writerow(['timestamp', 'passenger_count', 'timeofday', 'dayofweek'])
csvWriter.writerow(['datetime', 'int', 'int', 'string'])
csvWriter.writerow(['T', '', '', ''])
for i in range(len(ts_all)):
  csvWriter.writerow([seq.index[i], seq.passenger_count[i], seq.timeofday[i], seq.dayofweek[i]])
outputFile.close()