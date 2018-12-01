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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pprint
import numpy as np
import torch

# Need to run it from htmresearch top level:
# python projects/sdr_paper/pytorch_experiments/analyze_model.py model_path

if __name__ == '__main__':

  model = torch.load("projects/sdr_paper/pytorch_experiments/results/experiment10best/weight_sparsity0.40learning_rate0.040n500.0boost_strength1.0k50.0momentum0.250/model.pt")
  model.eval()
  print(model.l1.weight.data)

  dutyCycle = model.dutyCycle.numpy()
  dutyCycleSortedIndices = dutyCycle.argsort()[::-1]

  for i in range(20):
    w1 = model.l1.weight.data[dutyCycleSortedIndices[i]]
    w1 = w1.numpy().reshape((28, 28))
    plt.imshow(w1,clim=(0.0, 0.3))
    plt.colorbar()
    plt.savefig("temp"+str(i)+".jpg")
    plt.close()
