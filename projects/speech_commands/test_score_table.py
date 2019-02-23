# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import os

import numpy as np
from tabulate import tabulate

from htmresearch.frameworks.pytorch.sparse_speech_experiment import \
  SparseSpeechExperiment



def getErrorBars(expPath, suite):
  """
  Go through each experiment in the path. Get the best scores for each
  experiment whose hyperparameters were tuned based on accuracy on validation
  set. Return the overall mean, and stdev for test accuracy and noise accuracy.
  """

  # Get the iteration with maximum validation accuracy.
  max_scores = suite.get_histories_over_repetitions(exp=expPath,
                                                    tags=["testerror"],
                                                    aggregate=np.max)
  best = np.argmax(max_scores)

  # Compute the mean and std
  mean = suite.get_histories_over_repetitions(exp=expPath,
                                              tags=["testerror", "totalCorrect"],
                                              aggregate=np.mean)
  std = suite.get_histories_over_repetitions(exp=expPath,
                                             tags=["testerror", "totalCorrect"],
                                             aggregate=np.std)

  return {
    "test_score": (mean["testerror"][best], std["testerror"][best]),
    "noise_score": (mean["totalCorrect"][best], std["totalCorrect"][best])
  }



if __name__ == '__main__':
  suite = SparseSpeechExperiment()
  suite.parse_opt()
  suite.parse_cfg()
  experiments = suite.options.experiments or suite.cfgparser.sections()

  testScoresTable = [["Network", "Test Score", "Noise Score"]]
  for name in experiments:
    exps = suite.get_exps(suite.get_exp(name)[0])
    for exp in exps:
      if not os.path.exists(exp):
        continue

      errorBars = getErrorBars(exp, suite)
      test_score = u"{0:.2f} ± {1:.2f}".format(*errorBars["test_score"])
      noise_score = u"{0:,.0f} ± {1:.2f}".format(*errorBars["noise_score"])

      params = suite.get_params(exp=exp)
      testScoresTable.append([params["name"], test_score, noise_score])

  print(tabulate(testScoresTable, headers="firstrow", tablefmt="grid"))
