#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

from htmresearch.support.sp_paper_utils import *

expName = 'randomSDRVaryingSparsity'
plt.figure()
legendList = []
epochCheck = [0, 5, 10, 20, 40, 80]
for epoch in epochCheck:
  nrData = np.load \
    ('./results/input_output_overlap/{}_{}.npz'.format(expName, epoch))
  noiseLevelList =  nrData['arr_0']
  inputOverlapScore =  nrData['arr_1']
  outputOverlapScore = np.mean( nrData['arr_2'], 0)
  plt.plot(noiseLevelList, outputOverlapScore)
  legendList.append('epoch {}'.format(epoch))
plt.legend(legendList)
plt.xlabel('Noise Level')
plt.ylabel('Change of SP output')
plt.savefig('./figures/noise_robustness_{}.pdf'.format(expName))


expName = 'continuous_learning_without_topology'
plt.figure()
legendList = []
epochCheck = [79, 80, 219]
for epoch in epochCheck:
  nrData = np.load(
    './results/input_output_overlap/{}_{}.npz'.format(expName, epoch))
  noiseLevelList = nrData['arr_0']
  inputOverlapScore = nrData['arr_1']
  outputOverlapScore = np.mean(nrData['arr_2'], 0)
  plt.plot(noiseLevelList, outputOverlapScore)
  legendList.append('epoch {}'.format(epoch))
plt.legend(legendList)
plt.xlabel('Noise Level')
plt.ylabel('Change of SP output')
plt.savefig('./figures/noise_robustness_{}.pdf'.format(expName))