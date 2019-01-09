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
import numpy as np
import torch

def createValidationDataSampler(dataset, ratio):
  """
  Create `torch.utils.data.Sampler`s used to split the dataset into 2 ramdom
  sampled subsets. The first should used for training and the second for
  validation.

  :param dataset: A valid torch.utils.data.Dataset (i.e. torchvision.datasets.MNIST)
  :param ratio: The percentage of the dataset to be used for training. The
                remaining (1-ratio)% will be used for validation
  :return: tuple with 2 torch.utils.data.Sampler. (train, validate)
  """
  indices = np.random.permutation(len(dataset))
  training_count = int(len(indices) * ratio)
  train = torch.utils.data.SubsetRandomSampler(indices=indices[:training_count])
  validate = torch.utils.data.SubsetRandomSampler(indices=indices[training_count:])
  return (train, validate)

