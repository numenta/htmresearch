#!/usr/bin/env python
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

"""
This a python port of the matlab script 'sparsity_over_trial.m'
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from calculate_sparsity_over_trials import calculate_sparsity_over_trials
from get_response_mat import get_response_mat



def loadExperimentData(folder, area):
  """
  Loads the experiment's data from a MATLAB file into a python friendly structure.

  :param folder: Experiment data folder
  :param area: Experiament area to load. It should be 'V1' or 'AL'
  :return: The data as scipy matlab structure with the following fields:

      :Spiketrain.st: spike timing during stimulus (grating or naturalistic movie).
      :Spiketrain.st_gray: the spike timing during gray screen. The unit for
                           spike timing is sampling frame. And spike timing is
                           a 20X3 cell (corresponding to 20 repeats, and 3
                           stimulus (grating, and two naturalistic stimuli)).
                           The same for the spike timing of gray screen.
      :imgPara.stim_type: 3 type of stimulus, grating and two naturalistic stimuli.
      :imgPara.stim_time: the length of each stimulus is 32 sec.
      :imgPara.updatefr: the frame rate of stimulus on screen is 60 Hz.
      :imgPara.intertime: the time between two stimulus, or gray screen is 8 sec.
      :imgPara.dt: the sample rate, ~0.075
      :imgPara.F: number of sampling frames during stimulus is 32/0.075~=426
      :imgPara.F_gray: number of sampling frames during gray screen is 8/0.075~=106
      :ROI: the location of each neuron in the field.
  """
  filename = os.path.join(folder, "Combo3_{}.mat".format(area))
  contents = sio.loadmat(filename, variable_names=['data'],
                         struct_as_record=False, squeeze_me=True)
  return contents['data']



def getExperimentList():
  """
  Get list of folders containing the data of each experiment
  :return: list of folders
  """
  contents = sio.loadmat("./data/DataFolderList.mat",
                         variable_names=['DataFolderList'],
                         struct_as_record=False,
                         squeeze_me=True)

  return contents['DataFolderList']



if __name__ == "__main__":

  dataFolderList = getExperimentList()
  numActiveNeuron = 0
  numTotalNeuron = 0

  area = 'V1'  # area should be V1 or AL

  for stimType in [2]:
    for folder in dataFolderList:
      data = loadExperimentData(folder, area)
      spiketrain = data.spiketrain
      imgPara = data.imgPara
      date = folder[5:]

      numNeuron = spiketrain.size
      numFramesPerStim = int(round(imgPara.stim_time / imgPara.dt))
      gratingResponse = []
      # spikesPerNeuron records the accumulated spike counts for each
      # neuron during the 20 trials for each stimulus types
      spikesPerNeuron = np.zeros((numNeuron, 3))

      for i in xrange(numNeuron):
        for j in xrange(imgPara.stim_type):
          numSpike = 0
          for rep in xrange(imgPara.stimrep):
            spikesI = spiketrain[i].st[rep, j]
            numSpike = numSpike + len(np.atleast_1d(spikesI))

          spikesPerNeuron[i, j] = numSpike

      print "Number of cells: %d" % numNeuron
      # Population Response to Natural Stimuli
      goodCells = range(numNeuron)  # choose all the cells to be goodCells
      # goodCells = np.nonzero(spikesPerNeuron[:,stimType-1]>3)[0]
      spikeMat = get_response_mat(spiketrain, imgPara, stimType, goodCells, 0)

      # show sparsity over trials
      sparsity = calculate_sparsity_over_trials(spikeMat, imgPara)
      numActiveNeuron = numActiveNeuron + sparsity * len(goodCells) * numFramesPerStim
      numTotalNeuron = numTotalNeuron + len(goodCells)

      plt.figure()
      plt.subplot(2, 2, 1)
      plt.plot(sparsity, 'k')
      plt.xlabel('trial #')
      plt.ylabel('sparseness')
      plt.show()
      plt.savefig('figures/sparseness_over_time_stim_' + str(stimType)
                  + '_area_' + area + '_date_' + date + '.pdf')

    # plot the sparsity over trials
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(numActiveNeuron / numTotalNeuron / numFramesPerStim, '-o')
    plt.title('Area ' + area + ' Stimulus ' + str(stimType) + ' n=' +
              str(numTotalNeuron))
    plt.xlabel('Trial #')
    plt.ylabel('Sparseness')
    plt.show()
    plt.savefig('figures/sparsity/sparseness_over_time_stim_' + str(stimType)
                + '_area_' + area + '.pdf')
