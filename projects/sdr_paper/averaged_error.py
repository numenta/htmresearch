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

# This simulation computes false positive statistics over a wide range of n,
# sparsity, and the spiking threshold theta.

# Invoke with: time ipython --no-confirm-exit --gui=osx averaged_error.py

import numpy
from sympy import *
init_printing()
from IPython.display import display

from prettytable import PrettyTable
import math
import csv
import pickle


# The various symbols
oxp_s = Symbol("Omega_x'")
b_s = Symbol("b")
n_s = Symbol("n")
theta_s = Symbol("theta")
w_s = Symbol("w")
s_s = Symbol("s")
a_s = Symbol("a")

subsampledOmega = (binomial(s_s, b_s) * binomial(n_s - s_s, a_s - b_s)) / \
                  binomial(n_s, a_s)
subsampledFpF = Sum(subsampledOmega, (b_s, theta_s, s_s))
display(subsampledFpF)

subsampledOmegaSlow = (binomial(s_s, b_s) * binomial(n_s - s_s, a_s - b_s))
subsampledFpFSlow = Sum(subsampledOmegaSlow, (b_s, theta_s, s_s))/ binomial(
  n_s, a_s)
display(subsampledFpFSlow)

# Will hold the false positive rate for different values of theta
fpRate = {}

table = PrettyTable(["Sparsity", "n", "a", "s", "theta", "error rate"])

# Compute the various traces. Print out a table and put everything in a csv
# file.
with open("out.csv","w") as f:
  csvWriter = csv.writer(f)
  csvWriter.writerow( ["Sparsity", "n", "a", "s", "theta","error rate"])

  for theta in range(3,25,1):
    fpRate[theta] = []
    for n in range(10000,200000,20000):
      print "theta=",theta,"n=",n
      sparsity = 0.005
      while sparsity < 0.035:
        for s in range(20,51,5):
          a = round(sparsity*n)
          fp = subsampledFpFSlow.subs(s_s, s).subs(n_s, n).subs(a_s, a).subs(
            theta_s,theta).evalf()
          table.add_row([sparsity,n,a,s,theta,fp])
          csvWriter.writerow([sparsity,n,a,s,theta,fp])
          f.flush()
          fpRate[theta].append(fp)
        sparsity += 0.005

  print table.get_string().encode("utf-8")

  csvWriter.writerow([])
  csvWriter.writerow(["theta", "avgError","stdev", "min", "max", "median"])
  thetas = fpRate.keys()
  thetas.sort()
  for theta in thetas:
    # numpy.stdev seems to result in an error
    variance = numpy.var(fpRate[theta])
    csvWriter.writerow([theta,numpy.mean(fpRate[theta]),math.sqrt(variance),
                        numpy.min(fpRate[theta]), numpy.max(fpRate[theta]),
                        numpy.median(fpRate[theta])])

# Pickle the dict and the table
with open("fpRate.p","w") as pf:
  pickle.dump(fpRate, pf)

