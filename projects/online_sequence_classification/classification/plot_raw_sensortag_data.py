#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

import csv
import os
import matplotlib.pyplot as plt
from settings.acc_data import (INPUT_FILES, 
                               METRICS, 
                               DATA_DIR)
numRecordsToPlot = 500
plt.figure(figsize=(20, 13))
for inputFile in INPUT_FILES:
  filePath = os.path.join(DATA_DIR, inputFile)
  with open(filePath, 'rU') as f:
    reader = csv.reader(f)
    headers = reader.next()
    reader.next()
    t = []
    x = []
    y = []
    z = []

    for i, values in enumerate(reader):
      record = dict(zip(headers, values))

      try:
        for metric in METRICS:
          float(record[metric])
        x.append(float(record['x']))
        y.append(float(record['y']))
        z.append(float(record['z']))
        t.append(i)
      except ValueError:
        print "Not possible to convert some values of %s to a float" % record

      if i > numRecordsToPlot:
        break

    subplot_index = INPUT_FILES.index(inputFile)
    ax = plt.subplot(4, 1, subplot_index + 1)
    ax.plot(t, x, 'r', label='x')
    ax.plot(t, y, 'b', label='y')
    ax.plot(t, z, 'g', label='z')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.tight_layout()
    plt.title(inputFile)
    plt.xlim([0, numRecordsToPlot])
    plt.ylim([-8, 8])
    plt.xlabel('timestep')
    plt.ylabel('accelerometer')
    plt.grid()

plt.show()
