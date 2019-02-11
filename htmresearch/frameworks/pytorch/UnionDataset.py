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
import torch

from torch.utils.data import Dataset



class UnionDataset(Dataset):
  """
  Dataset used to create unions of two or more datasets. The union is created by
  applying the given transformation to the items in the dataset
  :param datasets: list of datasets of the same size to merge
  :param transform: function used to merge 2 items in the datasets
  """


  def __init__(self, datasets, transform):

    size = len(datasets[0])
    for ds in datasets:
      assert size == len(ds)

    self.datasets = datasets
    self.transform = transform


  def __getitem__(self, index):
    """
    Return the union value and labels for the item in all datasets
    :param index: The item to get
    :return: tuple with the merged data and labels associated with the data
    """
    union_data = None
    union_labels = []
    dtype = None
    device = None
    for i, ds in enumerate(self.datasets):
      data, label = ds[index]
      if i == 0:
        union_data = data
        dtype = label.dtype
        device = label.device
      else:
        union_data = self.transform(union_data, data)
      union_labels.append(label)

    return union_data, torch.tensor(union_labels, dtype=dtype, device=device)


  def __len__(self):
    return len(self.datasets[0])
