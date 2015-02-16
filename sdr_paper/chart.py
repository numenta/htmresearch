#!/usr/bin/env python
# ----------------------------------------------------------------------
# Copyright (C) 2015, Numenta Inc. All rights reserved.
#
# The information and source code contained herein is the
# exclusive property of Numenta Inc.  No part of this software
# may be used, reproduced, stored or distributed in any form,
# without explicit written authorization from Numenta Inc.
# ----------------------------------------------------------------------

"""TODO"""

import csv
import sys

from matplotlib import pyplot as plt



if __name__ == "__main__":
  if len(sys.argv) != 2:
    print "Must specify path argument"
  path = sys.argv[1]
  with open(path) as f:
    reader = csv.reader(f)
    n = None
    plotParamsList = []
    for i, row in enumerate(reader):
      if False and n and n != int(row[0]):
        for thetas, errors, label in plotParamsList:
          p = plt.plot(thetas, errors, label=label)
        plt.legend()
        plt.show()
        plotParamsList = []
      n = int(row[0])
      w = int(row[1])
      w_p = int(row[2])
      M = int(row[3])
      k = int(row[4])
      nTrials = int(row[5])
      errors = [float(e) for e in row[6:86]]

      thetas = [x+1 for x in xrange(len(errors))]
      label = "n=%i, w=%i, w'=%i" % (n, w, w_p)

      plotParamsList.append((thetas, errors, label))

    for thetas, errors, label in plotParamsList:
      #if "n=10000," in label:
      p = plt.plot(thetas, errors, label=label)

    plt.title("Calculated False Match Curves")
    plt.xlabel("Theta")
    plt.ylabel("False positive rate")
    plt.legend()
    plt.show()
