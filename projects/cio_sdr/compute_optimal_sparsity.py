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

# This simulation estimates optimal sparsity of CIO document fingerprints
# depending on word fingerprint sizes. We use the SDR calculations to estimate
# probability of false positives with random fingerprints.  The calculations
# assume an overlap matching followed by threshold.

# Invocation: ipython --no-confirm-exit --gui=osx compute_optimal_sparsity.py

import numpy
from sympy import *
init_printing()
from IPython.display import display

from prettytable import PrettyTable
import math
import csv
import pickle


# The various symbols
# n - overall number of bits in the retina
# s - number of ON bits in the word fingerprint or in a single semantic unit
# a - number of ON bits in the document fingerprint
# theta - overlap threshold

n_s = Symbol("n")
theta_s = Symbol("theta")
s_s = Symbol("s")
a_s = Symbol("a")
b_s = Symbol("b")

# Equations estimating error rate (two different versions)
subsampledOmega = (binomial(s_s, b_s) * binomial(n_s - s_s, a_s - b_s)) / \
                  binomial(n_s, a_s)
subsampledFpF = Sum(subsampledOmega, (b_s, theta_s, s_s))
display(subsampledFpF)

subsampledOmegaSlow = (binomial(s_s, b_s) * binomial(n_s - s_s, a_s - b_s))
subsampledFpFSlow = Sum(subsampledOmegaSlow, (b_s, theta_s, s_s))/ binomial(
  n_s, a_s)
display(subsampledFpFSlow)


def errorRate(s, a, theta=20, n=16384):
  fp = subsampledFpF.subs(s_s, s).subs(n_s, n).subs(a_s, a).subs(
    theta_s,theta).evalf()
  return fp

def errorRateSlow(s, a, theta=20, n=16384):
  fp = subsampledFpFSlow.subs(s_s, s).subs(n_s, n).subs(a_s, a).subs(
    theta_s,theta).evalf()
  return fp

def frange(x, y, step):
  while x < y:
    yield x
    x += step

headers = ["DocSparsity", "Doc FP bits", "Bits in semantic chunk",
                       "threshold","error rate"]
table = PrettyTable(headers)

# Compute the various error rates
# Print out a table and put everything in a csv file.
with open("sparsities.csv","w") as f:
  csvWriter = csv.writer(f)
  csvWriter.writerow(headers)

  # Normal retina uses about 2% sparsity per word
  n = 16384
  wordSparsity = round(0.01*n)

  # Small retina uses about 3% sparsity per word
  # n = 4096
  # wordSparsity = round(0.03*n)

  # Individual "chunks of meaning" are roughly 1/3 to 1/4 of the word
  s = round(wordSparsity/3)

  # Set theta to 50% or 25% to allow for lots of noise robustness
  for theta in [round(s/2)]: #, round(s/4)]:
    for docSparsity in frange(0.1,0.21,0.01):
      a = round(docSparsity*n)

      # print "sparsity=",docSparsity,"n=",n,"a=",a,"s=",s,"theta=",theta

      fp = subsampledFpFSlow.subs(s_s, s).subs(n_s, n).subs(a_s, a).subs(
        theta_s,theta).evalf()
      table.add_row([docSparsity,a,s,theta,fp])
      csvWriter.writerow([n,docSparsity,a,s,theta,fp])
      f.flush()

  print "Retina size=",n
  print table.get_string().encode("utf-8")

