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
import os
from collections import OrderedDict, defaultdict
from os.path import basename

import numpy as np

logging.basicConfig(level=logging.ERROR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from htmresearch.frameworks.pytorch.mnist_sparse_experiment import \
  MNISTSparseExperiment

NOISE_VALUES = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35",
                "0.4", "0.45", "0.5"]



def plotNoiseCurve(suite, values, results, plotPath, format):
  fig, ax = plt.subplots()
  fig.suptitle("Noise curve")
  ax.set_xlabel("Noise")
  ax.set_ylabel("Accuracy")
  for exp in results:
    values = suite.get_value(exp, 0, values, "last")
    df = pd.DataFrame.from_dict(values, orient='index')
    ax.plot(df["testerror"], **format[exp])

  plt.legend()
  plt.savefig(plotPath)
  plt.close()



def plotDropoutByTotalCorrect(results, plotPath, format):
  fig, ax = plt.subplots()
  fig.suptitle("Dropout by Total Correct")
  ax.set_xlabel("Dropout")
  ax.set_ylabel("Total Correct")
  for exp in results:
    data = OrderedDict(sorted(results[exp].items(), key=lambda x: x[0]))
    ax.plot(data.keys(), data.values(), **format[exp])

  xticks = data.keys()
  ax.xaxis.set_ticks(np.arange(min(xticks), max(xticks) + 0.1, 0.1))
  plt.legend()
  plt.savefig(plotPath)
  plt.close()



def filterResults(results, filter):
  """
  Filter results containing the given condition
  :param results: list of experiments returned by `suite.get_exps`
  :param filter: list of conditions on the experiment parameters. For example:
                 ["dropout0.0", "dropout0.50"]
  :return: filtered results
  """
  return [exp for exp in results if any(map(lambda v: v in exp, filter))]



def configureNoisePlot(suite, experiments, labels, linestyles, filter):
  """
  Load experiment results anc configure the "Noise curve" Plot
  :param suite: The configured experiment suite. Must call `parse_opt` and `
                parse_cfg` before calling this functions
  :param experiments: list containing the experiments to load
  :param experiments: list containing the experiments to load
  :param labels: list containing the plot labels for each experiment
  :param linestyles: list containing the plot linestyle for each experiment
  :param filter: list containing the specific parameters to filter
  :return: tuple containing the experiment results and plot formats to be passed
           to `plotNoiseCurve` function
  """
  formats = dict()
  results = []
  for i in xrange(len(experiments)):
    path = suite.get_exp(experiments[i])[0]
    data = suite.get_exps(path=path)
    data = filterResults(data, filter)

    # Format Linear Noise curve
    format = {exp: {
      "label": "{},{}".format(labels[i], basename(exp)),
      "linestyle": "{}".format(linestyles[i])
    } for exp in data}
    formats.update(format)
    results.extend(data)

  return (results, formats)



def configureDropoutByTotalCorrectPlot(suite, experiments, labels, linestyles):
  """
  Load experiment results anc configure the "Dropout By Total Correct" Plot
  :param suite: The configured experiment suite. Must call `parse_opt` and `
                parse_cfg` before calling this functions
  :param experiments: list containing the experiments to load
  :param labels: list containing the plot labels for each experiment
  :param linestyles: list containing the plot linestyle for each experiment
  :return: tuple containing the experiment results and plot formats to be passed
           to `plotDropoutByTotalCorrect` function
  """
  results = defaultdict(dict)
  formats = dict()
  for i in xrange(len(experiments)):
    experiment = experiments[i]
    formats[experiment] = {"label": labels[i], "linestyle": linestyles[i]}

    path = suite.get_exp(experiment)[0]
    data = suite.get_exps(path=path)
    for exp in data:
      dropout = suite.get_params(exp)["dropout"]
      totalCorrect = suite.get_value(exp, 0, "totalCorrect", "last")
      results[experiment][dropout] = totalCorrect

  return (results, formats)



if __name__ == '__main__':
  # Initialize experiment options and parameters
  suite = MNISTSparseExperiment()
  suite.parse_opt()
  suite.parse_cfg()
  path = suite.cfgparser.defaults()['path']

  # Plot Noise Curve (LinearNN)
  results, format = configureNoisePlot(suite,
                                       experiments=["DropoutExperimentDense",
                                                    "DropoutExperimentSparse"],
                                       labels=["dense", "sparse"],
                                       linestyles=["--", "-"],
                                       filter=["dropout0.0", "dropout0.50"])

  plotPath = os.path.join(path, "DropoutExperiment_accuracy.pdf")
  plotNoiseCurve(suite=suite, values=NOISE_VALUES, results=results, format=format,
                 plotPath=plotPath)

  # Plot Noise Curve (CNN)
  results, format = configureNoisePlot(suite,
                                       experiments=["DropoutExperimentDenseCNN",
                                                    "DropoutExperimentSparseCNN"],
                                       labels=["denseCNN", "sparseCNN"],
                                       linestyles=["--", "-"],
                                       filter=["dropout0.0", "dropout0.50"])

  plotPath = os.path.join(path, "DropoutExperimentCNN_accuracy.pdf")
  plotNoiseCurve(suite=suite, values=NOISE_VALUES, results=results, format=format,
                 plotPath=plotPath)

  # Plot Dropout by Noise (LinearNN)
  results, format = configureDropoutByTotalCorrectPlot(suite,
                                                       experiments=["DropoutExperimentDense",
                                                                    "DropoutExperimentSparse"],
                                                       labels=["Dense", "Sparse"],
                                                       linestyles=["--", "-"])

  plotPath = os.path.join(path, "DropoutExperiment_total_correct.pdf")
  plotDropoutByTotalCorrect(results=results, format=format, plotPath=plotPath)

  # Plot Dropout by Noise (CNN)
  results, format = configureDropoutByTotalCorrectPlot(suite,
                                                       experiments=["DropoutExperimentDenseCNN",
                                                                    "DropoutExperimentSparseCNN"],
                                                       labels=["DenseCNN", "SparseCNN"],
                                                       linestyles=["--", "-"])

  plotPath = os.path.join(path, "DropoutExperimentCNN_total_correct.pdf")
  plotDropoutByTotalCorrect(results=results, format=format, plotPath=plotPath)
