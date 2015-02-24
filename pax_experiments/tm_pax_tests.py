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
Created on Tue Feb 10 16:00:21 2015

@author: pfrady
"""

from nupic.encoders.sdrcategory import SDRCategoryEncoder
from pylab import *


import temporal_memory_pax as tmp

#%%
reload(tmp)


#%%

tm = tmp.TemporalMemory(150,15)


#%%
sdr_encoder = SDRCategoryEncoder(tm.columnDimensions, 22)

#%%


# This is basically the same function, except you pass in the values        
def plot_active_predicted(tm, active_cells=None, predicted_cells=None):
  '''plots the predicted and active cells'''
  
  if active_cells is None:
    active_cells = tm.activeCells
      
  if predicted_cells is None:
    predicted_cells = tm.predictiveCells
  
  active_unpredicted = setdiff1d(active_cells, predicted_cells)
  active_predicted = intersect1d(active_cells, predicted_cells)
  predicted_unactive = setdiff1d(predicted_cells, active_cells)
  
  col_idx = zeros(len(active_unpredicted))
  cell_idx = zeros(len(active_unpredicted))
  for i, cell in enumerate(active_unpredicted):
    col_idx[i] = tm.columnForCell(cell)
    cell_idx[i] = cell % tm.cellsPerColumn

  #plot(col_idx, cell_idx, 'o', c='y')
  
  col_idx = zeros(len(predicted_unactive))
  cell_idx = zeros(len(predicted_unactive))
  for i, cell in enumerate(predicted_unactive):
    col_idx[i] = tm.columnForCell(cell)
    cell_idx[i] = cell % tm.cellsPerColumn
      
  plot(col_idx, cell_idx, 'o', c='c')
  col_idx = zeros(len(active_predicted))
  cell_idx = zeros(len(active_predicted))
  for i, cell in enumerate(active_predicted):
    col_idx[i] = tm.columnForCell(cell)
    cell_idx[i] = cell % tm.cellsPerColumn
  
  u_cols = unique(col_idx)
  num_burst = zeros(len(u_cols))  
  for i, c in enumerate(u_cols):
    num_burst[i] = sum(c==col_idx)
          
  plot(col_idx, cell_idx, 'o', c='r')
  plot(u_cols, num_burst, 'r')  
  
#%% Lets start with a bunch of random SDRs

sdr_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

num_sequences = 5
num_sdr_per_seq = 4

sdr_dict = {}

for sdr in sdr_list:
  sdr_dict[sdr] = sdr_encoder.encode(sdr)
  

sequence_set = randint(0, len(sdr_list), (num_sequences, num_sdr_per_seq))
empty_sdr = zeros(sdr_dict[sdr_list[0]].shape)

#%% Run parameters

n_sequence_displays = 80 # number of times a sequence is shown
steps_between_sequences = 1 # number of empty sdrs between sequences

sequence_history = []
sdr_history = []
sequence_overlap = []

final_predicted_cells = []
final_active_cells = []
active_history = []
predicted_history= []

plot_state = False #True

p_flip_10 = 0 #0.4 # probability 1 flips to 0
p_flip_01 = 0 #0.02 # probability 0 flips to 1

#%% Run simulation

for i in range(n_sequence_displays):
  print i
  
  # First just show some empty sdrs
  for j in range(steps_between_sequences):
    tm.step(zeros(tm.columnDimensions))

    if plot_state:
      figure(1)
      clf()
      subplot(311)
      plot([0, tm.columnDimensions], [0,0], 'k')
      xlim([0, tm.columnDimensions])
      title('Empty')
        
      subplot(3,1, (2,3))
      plot(0,0)
      pause(0.01)
    
    #sdr_history.append(0)
      
  # Now show one of the sequences
  which_seq = randint(num_sequences)
  sequence_history.append(which_seq)
  
  for j in range(num_sdr_per_seq):
    sdr_name = sdr_list[sequence_set[which_seq,j]]
    sdr = sdr_dict[sdr_name]
    
    predicted_cells = tm.predictiveCells
    sdr_on_idxs = find(sdr)
    sdr_off_idxs = find(~sdr)
    flip_10 = rand(len(sdr_on_idxs)) < p_flip_10
    flip_01 = rand(len(sdr_off_idxs)) < p_flip_01
    
    sdr_add_idxs = sdr_off_idxs[flip_01]
    sdr_remove_idxs = sdr_on_idxs[flip_10]
    
    sdr_on_idxs = set(sdr_on_idxs)
    sdr_on_idxs = sdr_on_idxs - set(sdr_remove_idxs)
    sdr_on_idxs = sdr_on_idxs | set(sdr_add_idxs)
    
    (activeCells, burstCells) = tm.step(sdr, learn=True)
      
    if plot_state:
      figure(1)
      clf()
      subplot(311)
      plot(find(sdr), zeros(len(find(sdr))), '|k', ms=10)
      plot(sdr_add_idxs, zeros(len(sdr_add_idxs)), '|b', ms=10)
      plot(sdr_remove_idxs, zeros(len(sdr_remove_idxs)), '.r', ms=10)
      xlim([0, tm.columnDimensions])        
      title(sdr_name)
          
      subplot(3,1,(2,3))
      plot_active_predicted(tm)
      pause(0.01)
      
    active_history.append(find(activeCells))
    predicted_history.append(find(burstCells))
      
    sdr_history.append(sdr_name)
      
  # Just considering the last of the predictions
  final_predicted_cells.append(predicted_cells)
  final_active_cells.append(tm.activeCells)
  
  predicted_columns = set()
  for cell in predicted_cells:
      predicted_columns.add(tm.columnForCell(cell))
  active_columns = set()
  for cell in tm.activeCells:
      active_columns.add(tm.columnForCell(cell))
  
  sequence_overlap.append(len(predicted_columns & active_columns))
  #sequence_overlap.append(len(predicted_cells & tm.activeCells))



#%% Plot the spiking/prediction history for all presentations of a given sequence


for i in [0]: #range(num_sequences):
  figure('sdr history - ' + str(i))
  clf()

  sequence_idxs = find(array(sequence_history) == i)
  
  display_trials = arange(0, min(30, len(sequence_idxs)))
  
  num_display_trials = len(display_trials)
  
  for j, trial in enumerate(display_trials):
    for k in range(num_sdr_per_seq):
      print i,j,k
      subplot(num_sdr_per_seq, num_display_trials, j + num_display_trials * k + 1)
          
      active_idx = sequence_idxs[trial] *  num_sdr_per_seq + k           
        
      active_cells = active_history[active_idx]
      predicted_cells = predicted_history[active_idx]
          
      plot_active_predicted(tm, active_cells, predicted_cells)
      xticks([])
      if j > 0:
        yticks([])
      title(sdr_history[active_idx])

#%% Plot the error/overlap score

figure('sequence overlap')
clf()

#plot(sequence_overlap, '--k', lw=3)

colors = get_cmap('Set3', num_sequences)

max_sequence_displays = 0

sequence_overlap = array(sequence_overlap)
subplot(211)
for i in range(num_sequences):
  xvals = find(array(sequence_history)==i)
  plot(xvals, sequence_overlap[xvals] + i*0.1, lw=3, label='sequence ' + str(i), c=colors(i))
  
  if len(xvals) > max_sequence_displays:
    max_sequence_displays = len(xvals)

xticks(arange(len(sequence_history)), sequence_history)

if num_sequences < 10:
  legend(loc='upper left')

subplot(212)

overlap_exposures = nan * zeros((num_sequences, max_sequence_displays))

for i in range(num_sequences):
  xvals = find(array(sequence_history)==i)   
  plot(sequence_overlap[xvals] + i*0.1, lw=3, label='sequence ' + str(i), c=colors(i))
  
  overlap_exposures[i, :len(xvals)] = sequence_overlap[xvals]

plot(nanmean(overlap_exposures, 0), 'k', lw=4)


