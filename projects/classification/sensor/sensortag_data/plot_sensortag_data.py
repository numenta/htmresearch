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
import matplotlib.pyplot as plt

EXPERIMENTS = ['jump',
               'run',
               'sit',
               'stumble',
               'stairs-down',
               'stairs-up',
               'walk'
]

NUM_RECORDS_TO_PLOT = 80

plt.figure(figsize=(20, 10))
for exp in EXPERIMENTS:
  filePath = "data/%s-5min.csv" % exp
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
      t.append(i)
      x.append(record['x'])
      y.append(record['y'])
      z.append(record['z'])
      if i > NUM_RECORDS_TO_PLOT:
        break
        
    subplot_index = EXPERIMENTS.index(exp)
    plt.subplot(4, 2, subplot_index + 1)
    plt.plot(t, x, 'r', t, y, 'b', t, z, 'g')
    plt.tight_layout()
    plt.title(exp)
    plt.xlim([0, NUM_RECORDS_TO_PLOT])
    plt.ylim([-8, 8])
    plt.xlabel('timestep')
    plt.ylabel('accelerometer')
    plt.grid()

plt.show()

