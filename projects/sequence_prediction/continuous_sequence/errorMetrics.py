# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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


def NRMSE(data, pred):
  return np.sqrt(np.nanmean(np.square(pred-data)))/\
         np.nanstd(data)



def NRMSE_sliding(data, pred, windowSize):
  """
  Computing NRMSE in a sliding window
  :param data:
  :param pred:
  :param windowSize:
  :return: (window_center, NRMSE)
  """

  halfWindowSize = int(round(float(windowSize)/2))
  window_center = range(halfWindowSize, len(data)-halfWindowSize, int(round(float(halfWindowSize)/5.0)))
  nrmse = []
  for wc in window_center:
    nrmse.append(NRMSE(data[wc-halfWindowSize:wc+halfWindowSize],
                       pred[wc-halfWindowSize:wc+halfWindowSize]))

  return (window_center, nrmse)


def altMAPE(groundTruth, prediction):
  error = abs(groundTruth - prediction)
  altMAPE = 100.0 * np.sum(error) / np.sum(abs(groundTruth))
  return altMAPE


def MAPE(groundTruth, prediction):
  MAPE = np.nanmean(
    np.abs(groundTruth - prediction)) / np.nanmean(np.abs(groundTruth))
  return MAPE