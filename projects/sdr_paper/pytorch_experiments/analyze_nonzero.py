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
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from htmresearch.frameworks.pytorch.benchmark_utils import register_nonzero_counter, unregister_counter_nonzero

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
  :type results: list[string]
  :param filter: list of conditions on the experiment parameters. For example:
                 ["dropout0.0", "dropout0.50"]
  :type filter: list[string] or None
  :return: filtered results
  :rtype: list[string]
 """
  return [exp for exp in results if any(map(lambda v: v in exp, filter))]



def evaluateModel(model, loader, progess=None):
  """
  Evaluate pre-trained model using given test dataset loader

  :param model: Pretrained model
  :type model: torch.nn.Module
  :param loader: test dataset loader
  :type loader: :class:`torch.utils.data.DataLoader`
  :param progess: Optional progress bar description. None for no progress bar
  :type progess: string or None
  :return: dictionary with "accuracy", "loss", "num_correct"
  :rtype: dict
  """
  model.eval()
  loss = 0
  correct = 0
  dataset_len = len(loader.sampler)

  if progess is not None:
    loader = tqdm(loader, desc=progess)

  with torch.no_grad():
    for data, target in loader:
      output = model(data)
      loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  loss /= dataset_len
  accuracy = 100. * correct / dataset_len

  return {"num_correct": correct,
          "loss": loss,
          "accuracy": accuracy}



if __name__ == '__main__':
  # Initialize experiment options and parameters
  suite = MNISTSparseExperiment()
  suite.parse_opt()
  suite.parse_cfg()
  experiments = suite.options.experiments or suite.cfgparser.sections()

  # Dataset transformations used during training. See mnist_sparse_experiment.py
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

  # Optional result filter
  results_filter = None  # ["min_weight0.0", "min_weight0.1"])

  for expName in experiments:
    path = suite.get_exp(expName)[0]
    results = suite.get_exps(path=path)

    if results_filter is not None:
      results = filterResults(results, results_filter)

    for exp in results:
      print()
      print(exp)
      params = suite.get_params(exp)

      # Initialize MNIST test dataset for this experiment
      test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(params["datadir"], train=False, download=True,
                       transform=transform),
        batch_size=params["test_batch_size"], shuffle=True)

      # Load pre-trained model and evaluate with test dataset
      model = torch.load(os.path.join(exp, "model.pt"))

      tables = []
      for ratio in np.linspace(0.0, 0.1, 11):
        label = str(ratio)
        model.pruneWeights(ratio)

        # Collect nonzero
        nonzero = {}
        register_nonzero_counter(model, nonzero)
        eval = evaluateModel(model, test_loader, progess=label)
        unregister_counter_nonzero(model)

        # Create table with results
        table = pd.DataFrame.from_dict(nonzero)
        table = table.assign(**eval)

        # Filter results for the 'weight' variable only
        tables.append(pd.DataFrame({label: table.xs("weight")}))

      merged = pd.concat(tables, axis=1)
      merged.drop(["input", "output"], inplace=True)
      merged.dropna(inplace=True)

      # Print results
      print(tabulate(merged, headers='keys', tablefmt='fancy_grid'))
