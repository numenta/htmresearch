# ------------------------------------------------
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
"""
Analyze model unions by testing a previously MNIST trained model with two
MNIST training datasets overlaid on top of each other. Compute multi-label
accuracy using "Exact match" metric.
"""
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from htmresearch.frameworks.pytorch.dataset_utils import UnionDataset
from htmresearch.frameworks.pytorch.mnist_sparse_experiment import MNISTSparseExperiment
from tabulate import tabulate
from torchvision import datasets, transforms
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR)



def create_union_mnist_dataset():
  """
  Create a UnionDataset composed of two versions of the MNIST datasets
  where each item in the dataset contains 2 distinct images superimposed
  """
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

  mnist1 = datasets.MNIST('data', train=False, download=True, transform=transform)
  data1 = zip(mnist1.test_data, mnist1.test_labels)

  # Randomize second dataset
  mnist2 = datasets.MNIST('data', train=False, download=True, transform=transform)
  data2 = zip(mnist2.test_data, mnist2.test_labels)
  random.shuffle(data2)

  # Reorder images of second dataset with same label as first dataset
  for i in range(len(data2)):
    if data1[i][1] == data2[i][1]:
      # Swap indices with same label to a location with diffent label
      for j in range(len(data1)):
        if data1[i][1] != data2[j][1] and data2[i][1] != data1[j][1]:
          swap = data2[j]
          data2[j] = data2[i]
          data2[i] = swap
          break

  # Update second dataset with new item order
  mnist2.test_data, mnist2.test_labels = zip(*data2)

  # Combine the images of both datasets using the maximum value for each pixel
  return UnionDataset(datasets=[mnist1, mnist2],
                      transform=lambda x, y: torch.max(x, y))



def exact_match(pred, target):
  """
  Compute "Exact match" metric, also called "Subset accuracy" indicating the
  number of samples that have all their labels classified correctly.
  See https://en.wikipedia.org/wiki/Multi-label_classification

  :param pred: Predicted labels
  :param target: Correct labels
  :return: containing a 1 at the prediction where all labels in the prediction
           match all labels in the target
  """
  res = torch.eq(target.sort(dim=1)[0], pred.sort(dim=1)[0])
  return res.prod(dim=1)



def evaluate(model, loader, device):
  correct = 0
  dataset_len = len(loader.sampler)

  with torch.no_grad():
    for data, target in loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      pred = torch.topk(output, k=target.size(1), dim=1)[1]
      res = exact_match(pred, target)
      correct += res.sum().item()

  return {"total_correct": correct,
          "accuracy": float(correct) / dataset_len}



def random_model(data):
  return torch.rand(data.size(0), 10)



def main():
  # Set fixed seed. Only works on cpu
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)

  dataset = create_union_mnist_dataset()

  # Load experiment
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  suite = MNISTSparseExperiment()
  suite.parse_opt()
  suite.parse_cfg()
  experiments = suite.options.experiments or suite.cfgparser.sections()
  table = {}
  progress = tqdm(experiments)
  for expName in progress:
    progress.set_description(expName)
    path = suite.get_exp(expName)[0]
    results = suite.get_exps(path=path)
    for exp in results:
      model_file = os.path.join(exp, "model.pt")
      if os.path.exists(model_file):
        model = torch.load(model_file, map_location=device)
        params = suite.get_params(exp)
        test_loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                                  batch_size=params["test_batch_size"])
        table[params['name']] = evaluate(model=model, loader=test_loader, device=device)

  # Random model
  test_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=4)
  table["random"] = evaluate(model=random_model, loader=test_loader, device=device)

  # Save results
  df = pd.DataFrame.from_dict(table)
  df.to_csv("union_experiment_results.csv")
  print(tabulate(df, tablefmt='fancy_grid', headers='keys', numalign="right"))

  # Plot first 100 images in the dataset
  fig = plt.figure(figsize=(10, 10))
  for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1)
    ax.set_axis_off()
    img, label = dataset[i]
    ax.imshow(img.numpy().reshape((28, 28)), cmap='gray')
    ax.set_title(str(label.numpy()))
  plt.tight_layout()
  plt.savefig("union_experiment_sample_images.png")
  plt.close()



if __name__ == '__main__':
  main()
