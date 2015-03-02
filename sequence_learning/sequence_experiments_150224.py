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
sequence_params['total_displays'] = 1000
sequence_params['num_elements'] = 1000
sequence_params['num_sequences'] = 10
sequence_params['sequence_length'] = 8
sequence_params['sequence_order'] = 0
sequence_params['p_rand_sdr'] = 0
sequence_params['force_rand_sdr'] = True
sequence_params['unique_rand_sdr'] = True

num_runs = 8
filenames = []

for i in range(num_runs):
  
  tm = TemporalMemory(minThreshold=30, activationThreshold=30,  maxNewSynapseCount=40)
  
  filename = squ.runTMSequenceExperiment(tm, sequence_params)

  filenames.append(filename)  


#%%
reference_file = ('filenames-sl%02d-ho%01d-ne%04d-ns%02d-n'
                  % (sequence_params['sequence_length'], sequence_params['sequence_order'],
                     sequence_params['num_elements'], sequence_params['num_sequences'])
                  + datetime.date.today().strftime('%y%m%d') + '.npz')

savez_compressed(reference_file, filenames=filenames)


#%%

#reference_file = 'filenames-sl2-ho0-ne1000-ns100-n150227.npz'

with load(reference_file) as data:
  filenames = data['filenames']
  
  
sequence_error_all = zeros((len(filenames), 1))
for i, filename in enumerate(filenames):
  with load(filename) as data:
    sequence_error = data['sequence_error']

  if len(sequence_error) > sequence_error_all.shape[1]:
    sequence_error_all.resize((len(filenames), len(sequence_error)))
  
  sequence_error_all[i,:len(sequence_error)] = sequence_error
  

figure('Sequence error several runs')
clf()
plot(mean(sequence_error_all, 0))

smooth_factor = 10
plot(convolve(mean(sequence_error_all, 0), ones(smooth_factor)/smooth_factor, mode='same'), 'k', lw=3)
  