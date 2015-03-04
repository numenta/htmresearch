#!/usr/bin/env python

# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
This module runs many sequence experiments and creates plots.

Created on Wed Feb 25 13:27:55 2015

@author: pfrady
"""

from __future__ import division

from pylab import *

from nupic.research.temporal_memory import TemporalMemory

import ho_sequence_utils as squ

import datetime

#%%
sequence_params = {}
sequence_params['totalDisplays'] = 4000
sequence_params['numElements'] = 1000
sequence_params['numSequences'] = 10
sequence_params['sequenceLength'] = 8
sequence_params['sequenceOrder'] = 2
sequence_params['pRandSDR'] = 0
sequence_params['forceRandSDR'] = True
sequence_params['uniqueRandSDR'] = True

num_runs = 8
filenames = []

for i in range(num_runs):
  
  tm = TemporalMemory(minThreshold=30, activationThreshold=30,  maxNewSynapseCount=40)
  
  filename = squ.runTMSequenceExperiment(tm, sequence_params)

  filenames.append(filename)  


#%%
reference_file = ('filenames-sl%02d-ho%01d-ne%04d-ns%02d-n'
                  % (sequence_params['sequenceLength'], sequence_params['sequenceOrder'],
                     sequence_params['numElements'], sequence_params['numSequences'])
                  + datetime.date.today().strftime('%y%m%d') + '.npz')

savez_compressed(reference_file, filenames=filenames)


#%% First order sequence learning with longer sequences

# These are the simulation files for the data in sequence_memory_random.pptx
reference_files = []
reference_files.append('filenames-sl02-ho0-ne1000-ns10-n150302.npz')
reference_files.append('filenames-sl04-ho0-ne1000-ns10-n150302.npz')
reference_files.append('filenames-sl08-ho0-ne1000-ns10-n150302.npz')
reference_files.append('filenames-sl16-ho0-ne1000-ns10-n150302.npz')

classLabels = ('Sequence Length: 2', 'Sequence Length: 4', 
               'Sequence Length: 8', 'Sequence Length: 16')

#%% HO sequence learning with more order

# These are the simulation files for the data in sequence_memory_random.pptx
reference_files = []
reference_files.append('filenames-sl08-ho0-ne1000-ns10-n150302.npz')
reference_files.append('filenames-sl08-ho1-ne1000-ns10-n150302.npz')
reference_files.append('filenames-sl08-ho2-ne1000-ns10-n150302.npz')
reference_files.append('filenames-sl08-ho3-ne1000-ns10-n150302.npz')
reference_files.append('filenames-sl08-ho6-ne1000-ns10-n150302.npz')

classLabels = ('Sequence Order: 0', 'Sequence Order: 1', 'Sequence Order: 2', 
               'Sequence Order: 3', 'Sequence Order: 6')


#%% Set up the reference files and class names for the multi plots

filenames = []
runClasses = []
for i, ref_file in enumerate(reference_files):
  
  with load(ref_file) as data:
    fns = data['filenames']
    filenames.extend(fns)
    runClasses.extend(i * ones(len(fns)))


#%% Plot several simulations averaged together
  
squ.plotTMErrorAllMulti(filenames, runClasses, smoothFactor=30, classLabels=classLabels)

legend()
#title('High Order Sequence Learning')


#%%

squ.plotTMErrorSequenceMulti(filenames, runClasses, classLabels=classLabels)

legend()
#title('First Order, All Predictable Elements')
title('First Order, All Predictable Elements')

#%%

squ.plotTMErrorHOMulti(filenames, runClasses, classLabels=classLabels, smoothFactor=10)

legend()
title('High Order Learning, HO Element')