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
import multiprocessing
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from htmresearch.frameworks.pytorch.benchmark_utils import (
  register_nonzero_counter, unregister_counter_nonzero)
from htmresearch.frameworks.pytorch.image_transforms import RandomNoise
from htmresearch.frameworks.pytorch.model_utils import evaluateModel

logging.basicConfig(level=logging.ERROR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import pandas as pd
from htmresearch.frameworks.pytorch.mnist_sparse_experiment import \
  MNISTSparseExperiment



def filterResults(results, filter):
  """
  Filter results containing the given condition

  :param results: list of experiments returned by `suite.get_exps`
  :type results: list[string]
  :param filter: list of conditions on the experiment parameters. For example:
                 ["dropout0.0", "dropout0.50"]
  :type filter: list[string] or None
  :return: filtered results
  :rtype: list[string]
 """
  return [exp for exp in results if any(map(lambda v: v in exp, filter))]



def analyzeWeightPruning(args):
  """
  Multiprocess function used to analyze the impact of nonzeros and accuracy
  after pruning weights of a pre-trained model.

  :param args:  tuple with the following arguments: (experiment path,
                configuration parameters, minWeight, progress bar position)
  :type args:   tuple

  :return: Panda DataFrame with the nonzero count for every weight variable in
           the model and the evaluation results after the pruning the weights.
  :rtype: :class:`pandas.DataFrame`
  """
  path, params, minWeight, position = args

  # Dataset transformations used during training. See mnist_sparse_experiment.py
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

  # Initialize MNIST test dataset for this experiment
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(params["datadir"], train=False, download=True,
                   transform=transform),
    batch_size=params["test_batch_size"], shuffle=True)

  # Load pre-trained model and evaluate with test dataset
  model = torch.load(os.path.join(path, "model.pt"), map_location="cpu")

  tables = []
  label = str(minWeight)
  name = params["name"]
  desc = "{}.min({})".format(name, minWeight)
  model.pruneWeights(minWeight)

  # Collect nonzero
  nonzero = {}
  register_nonzero_counter(model, nonzero)
  results = evaluateModel(model, test_loader, {"desc": desc, "position": position})
  unregister_counter_nonzero(model)

  # Create table with results
  table = pd.DataFrame.from_dict(nonzero)
  table = table.assign(noise_score=results["total_correct"])
  table = table.assign(accuracy=results["accuracy"])

  # Filter result for the 'weight' variable only
  tables.append(pd.DataFrame({label: table.xs("weight")}))

  # Compute noise score
  noise_values = tqdm([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                      position=position)
  for noise in noise_values:
    noise_values.set_description("{}.noise({})".format(desc, noise))

    # Add noise to dataset transforms
    transform.transforms.append(
      RandomNoise(noise, whiteValue=0.1307 + 2 * 0.3081))

    # Evaluate model with noise
    results = evaluateModel(model, test_loader)

    # Remove noise from dataset transforms
    transform.transforms.pop()

    # Update noise score
    table["noise_score"] += results["total_correct"]

  merged = pd.concat(tables, axis=1)
  merged.drop(["input", "output"], inplace=True)
  merged.dropna(inplace=True)
  return merged



def plotDataframe(table, title, plotPath):
  """
  Plot Panda dataframe.

  :param table: Panda dataframe returned by :func:`analyzeWeightPruning`
  :type table: :class:`pandas.DataFrame`
  :param title: Plot title
  :type title: str
  :param plotPath: Plot full path
  :type plotPath: str

  """
  plt.figure()
  axes = table.T.plot(subplots=True, sharex=True, grid=True, legend=True,
                      title=title, figsize=(8, 11))

  # Use fixed scale for "accuracy"
  accuracy = next(ax for ax in axes if ax.lines[0].get_label() == 'accuracy')
  accuracy.set_ylim(0.0, 1.0)

  plt.savefig(plotPath)
  plt.close()



def main():
  # Initialize experiment options and parameters
  suite = MNISTSparseExperiment()
  suite.parse_opt()
  suite.parse_cfg()
  experiments = suite.options.experiments or suite.cfgparser.sections()

  # Optional result filter
  results_filter = None  # ["min_weight0.0", "min_weight0.1"])

  args = []
  for expName in experiments:
    path = suite.get_exp(expName)[0]
    results = suite.get_exps(path=path)

    if results_filter is not None:
      results = filterResults(results, results_filter)

    for exp in results:
      for i, minWeight in enumerate(np.linspace(0.0, 0.1, 21)):
        args.append([exp, suite.get_params(exp), minWeight, i])

    pool = multiprocessing.Pool()

    # Analyze weight pruning
    tables = pool.map(analyzeWeightPruning, args)
    merged = pd.concat(tables, axis=1).sort_index(axis=1)
    plotname = "weight_pruning_{}".format(expName)
    plotDataframe(merged, plotname, "{}.pdf".format(plotname))
    print()
    print(plotname)
    print(tabulate(merged, headers='keys', tablefmt='fancy_grid',
                   numalign="right"))




if __name__ == '__main__':
  main()
