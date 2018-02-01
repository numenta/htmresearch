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
import numpy as np

def get_response_mat(spiketrain, imgPara, stimType, goodCells, plotRaster):
    
    numNeuron = len(goodCells)  # the number of neurons
    # imgPara.stim_time = 32s, imgPara.dt = 0.075103, 
    # numFramesPerStim is the number of the frames within 32s movie stimulus
    numFramesPerStim = int(round(imgPara.stim_time / imgPara.dt))

    spikeMat = []
    # generate the spike timing for all the neurons through all trials
    for rep in xrange(imgPara.stimrep):
        spikesCurrentTrial = np.zeros((numNeuron, numFramesPerStim))
        spikesRaster = []
        cellI = 0
        for i in goodCells:
            # spikesI: spiking timing of a specific neuron at a specific trial
            spikesI = np.atleast_1d(spiketrain[i].st[rep, stimType - 1])
            spikesI = np.round(spikesI[np.nonzero(spikesI <= numFramesPerStim)])
            spikesI = spikesI[np.nonzero(spikesI > 0)]
            spikesI = spikesI.astype(int)
            spikesI = spikesI - 1

            # along the 426 frames, spike timings was labeled
            spikesCurrentTrial[cellI, spikesI] = 1
            cellI = cellI + 1
            spikesRaster.append(spikesI * imgPara.dt - 1)
        

        # return spikeMat as the spiking time for all neurons
        spikeMat.append(spikesCurrentTrial)  
    
    # change spikeMat to be numpy array
    spikeMat = np.array(spikeMat)   
    return spikeMat
