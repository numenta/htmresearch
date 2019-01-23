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
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import logging

logging.basicConfig(level=logging.ERROR)

import matplotlib

matplotlib.use("Agg")
from tabulate import tabulate

import pandas as pd
from htmresearch.frameworks.pytorch.mnist_sparse_experiment import \
  MNISTSparseExperiment



def filterResults(results, filter):
  """
  Filter results containing the given condition
  :param results: list of experiments returned by `suite.get_exps`
  :param filter: list of conditions on the experiment parameters. For example:
                 ["dropout0.0", "dropout0.50"]
  :return: filtered results
  """
  return [exp for exp in results if any(map(lambda v: v in exp, filter))]



if __name__ == '__main__':
  # Initialize experiment options and parameters
  suite = MNISTSparseExperiment()
  suite.parse_opt()
  suite.parse_cfg()
  columns = None  # ['linearSdr.linearSdr1', 'linearSdr.linearSdr1.l1']
  if suite.options.experiments is not None:
    for expName in suite.options.experiments:
      path = suite.get_exp(expName)[0]
      data = suite.get_exps(path=path)
      data = filterResults(data, ["min_weight0.0min_dutycycle0.0", "min_weight0.10"])
      for exp in data:
        values = suite.get_value(exp, 0, "nonzeros", "last")
        df = pd.DataFrame.from_dict(values)
        print()
        print(exp)
        if columns is not None:
          df = df[columns]

        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
  else:
    print("Failed to read experiments from arguments.",
          "Use '-e' to select experiments or '--help' for other options.")