# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

from __future__ import print_function

import pprint
import numpy as np

from htmresearch.frameworks.pytorch.mnist_sparse_experiment import \
  MNISTSparseExperiment


if __name__ == '__main__':

  expName = "./results/experiment1"

  suite = MNISTSparseExperiment()

  experiments = suite.get_exps(expName)

  expParams = suite.get_params(expName)
  pprint.pprint(expParams)

  # Retrieve the last totalCorrect from each experiment
  # Print them sorted from best to worst
  values, params = suite.get_values_fix_params(
          expName, 0, "totalCorrect", "last")
  v = np.array(values)
  sortedIndices = v.argsort()
  for i in sortedIndices[::-1]:
    print(v[i],params[i]["name"])

  print()

  for boost in expParams ["boost_strength"]:
    # Retrieve the last totalCorrect from each experiment with boost 0
    # Print them sorted from best to worst
    values, params = suite.get_values_fix_params(
            expName, 0, "totalCorrect", "last", boost_strength=boost)
    v = np.array(values)
    sortedIndices = v.argsort()
    print("Average with boost", boost,"=",v.mean())
    # for i in sortedIndices[::-1]:
    #   print(v[i],params[i]["name"])


  for b in expParams ["k"]:
    # Retrieve the last totalCorrect from each experiment with boost 0
    # Print them sorted from best to worst
    values, params = suite.get_values_fix_params(
            expName, 0, "totalCorrect", "last", k=b)
    v = np.array(values)
    sortedIndices = v.argsort()
    print("Average with k", b,"=",v.mean())
    # for i in sortedIndices[::-1]:
    #   print(v[i],params[i]["name"])

