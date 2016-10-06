# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
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

"""
Script to analyze trauma experiment result
"""

if __name__ == "__main__":
  plt.ion()
  expName = 'trauma_boosting_with_topology'
  numEpochs = 600
  killAt = 150

  inputCoverageTraumaRegion = []
  inputCoverageControlRegion = []
  inputCoverageBeforeTrauma = None
  inputCoverageAfterTrauma = None
  inputCoverageAfterRecovery = None
  epochList = []
  for epoch in range(99, 500):
    epochList.append(epoch)
    inputSpaceCoverage = np.load \
      ('./results/InputCoverage/{}_{}.npz'.format(expName, epoch))
    inputSpaceCoverage = inputSpaceCoverage['arr_0']
    inputCoverageTraumaRegion.append(
      np.mean(inputSpaceCoverage[14:18, 14:18]))

    inputCoverageControlRegion.append(
      np.mean(inputSpaceCoverage[27:31, 8:12]))

    if epoch == killAt - 1:
      inputCoverageBeforeTrauma = inputSpaceCoverage
    if epoch == killAt:
      inputCoverageAfterTrauma = inputSpaceCoverage
    if epoch == numEpochs-1:
      inputCoverageAfterRecovery = inputSpaceCoverage

  # Plot coverage factor for trauma region and a control region
  fig, axs = plt.subplots(2, 2)
  axs[0, 0].plot(epochList, inputCoverageControlRegion, label='control')
  axs[0, 0].plot(epochList, inputCoverageTraumaRegion, label='trauma')
  # axs[0, 0].set_xlim([killAt-50, numEpochs])
  axs[0, 0].set_xlim([100, 500])
  axs[0, 0].set_ylim([15, 50])
  axs[0, 0].set_xlabel('Time')
  # axs[0, 0].set_aspect('equal')
  plt.legend()
  plt.savefig('figures/traumaRecovery.pdf')

  # Compare before and after recovery
  fig, axs = plt.subplots(2, 2)
  im = axs[0, 0].pcolor(
    inputCoverageAfterRecovery.astype('float32') - inputCoverageAfterTrauma,
  vmin=-12, vmax=12)
  axs[0, 0].set_title('Synapse Growth')
  axs[0, 0].set_xlim([-1, 33])
  axs[0, 0].set_ylim([-1, 33])
  axs[0, 0].set_aspect('equal')
  plt.axis('equal')
  cax = fig.add_axes([0.15, 0.4, 0.3, 0.05])
  fig.colorbar(im, cax=cax, orientation='horizontal',
               ticks=[-10, -5, 0, 5, 10])
  plt.savefig('figures/synapseGrowthAfterTrauma.pdf')

  # plot RFcenters and inputCoverage over training
  for epoch in range(killAt-50, numEpochs):
    fig, axs = plt.subplots(2, 2)
    RFcenterInfo = np.load('./results/RFcenters/{}_{}.npz'.format(expName, epoch))
    RFcenters = RFcenterInfo['arr_0']
    avgDistToCenter = RFcenterInfo['arr_1']
    axs[0, 1].scatter(RFcenters[:, 0], RFcenters[:, 1], s=4, c=[1,1,1])
    axs[0, 1].set_title('RF centers')
    axs[0, 1].set_xlim([-1, 33])
    axs[0, 1].set_ylim([-1, 33])
    axs[0, 1].set_aspect('equal')
    plt.axis('equal')
    inputSpaceCoverage = np.load \
      ('./results/InputCoverage/{}_{}.npz'.format(expName, epoch))
    inputSpaceCoverage = inputSpaceCoverage['arr_0']

    im = axs[0, 0].pcolor(inputSpaceCoverage, vmin=10, vmax=80)
    axs[0, 0].set_title('Input Space Coverage')
    axs[0, 0].set_xlim([-1, 33])
    axs[0, 0].set_ylim([-1, 33])
    axs[0, 0].set_aspect('equal')
    plt.axis('equal')
    cax = fig.add_axes([0.15, 0.4, 0.3, 0.05])
    fig.colorbar(im, cax=cax, orientation='horizontal',
                 ticks=[0, 20, 40, 60, 80])

    fig.delaxes(axs[1, 0])
    fig.delaxes(axs[1, 1])
    plt.savefig('figures/traumaMovie/frame_{}.png'.format(epoch))
    plt.close(fig)
