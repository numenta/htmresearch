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
Created on Thu Jan 29 16:18:44 2015

This file runs sequence memory experiments. A set of random sequences are
created from a library of SDRs. The sequences are presented to the temporal 
memory in a random order, and it learns the sequences. Because of the limited
dictionary of SDRs, most sequences created will be "high-order", in that 
different sequences will contain the same pieces.

@author: pfrady
"""

from pylab import *

from nupic.research.temporal_memory import TemporalMemory
from nupic.encoders.sdrcategory import SDRCategoryEncoder


tm = TemporalMemory()

#%%
sdr_encoder = SDRCategoryEncoder(tm.numberOfColumns(), 40)


#%%

# This function actually doesn't work, because predictiveCells is for next step
def plot_tm_state(tm):
  ''' plots the state of a temporal memory'''
  
  plot(-1,-1, 'o', c='c', label='Active Cell')
  plot(-1,-1, 'o', c='r', label='Predictive Cell')    
  
  col_idx = zeros(len(tm.activeCells))
  cell_idx = zeros(len(tm.activeCells))        
  for i, cell in enumerate(tm.activeCells):
    col_idx[i] = tm.columnForCell(cell)
    cell_idx[i] = cell % tm.cellsPerColumn
      
  plot(col_idx, cell_idx, 'o', c='y')
  
  col_idx = zeros(len(tm.predictiveCells))
  cell_idx = zeros(len(tm.predictiveCells))
  for i, cell in enumerate(tm.predictiveCells):
    col_idx[i] = tm.columnForCell(cell)
    cell_idx[i] = cell % tm.cellsPerColumn
      
  plot(col_idx, cell_idx, 'o', c='r')
      
  axis([0, tm.numberOfColumns(), 0, tm.cellsPerColumn])

# This is basically the same function, except you pass in the values        
def plot_active_predicted(tm, active_cells=None, predicted_cells=None):
  '''plots the predicted and active cells'''
  
  if active_cells is None:
    active_cells = tm.activeCells
      
  if predicted_cells is None:
    predicted_cells = tm.predictiveCells
  
  active_unpredicted = active_cells - predicted_cells
  active_predicted = active_cells & predicted_cells
  predicted_unactive = predicted_cells - active_cells
  
  col_idx = zeros(len(active_unpredicted))
  cell_idx = zeros(len(active_unpredicted))
  for i, cell in enumerate(active_unpredicted):
    col_idx[i] = tm.columnForCell(cell)
    cell_idx[i] = cell % tm.cellsPerColumn

  plot(col_idx, cell_idx, 'o', c='y')
  
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
      
  plot(col_idx, cell_idx, 'o', c='r')

  
  
  

#%% Lets start with a bunch of random SDRs

sdr_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

num_sequences = 40
num_sdr_per_seq = 6

sdr_dict = {}

for sdr in sdr_list:
  sdr_dict[sdr] = sdr_encoder.encode(sdr)
  

sequence_set = randint(0, len(sdr_list), (num_sequences, num_sdr_per_seq))
empty_sdr = zeros(sdr_dict[sdr_list[0]].shape)

#%% Run parameters

n_sequence_displays = 800 # number of times a sequence is shown
steps_between_sequences = 1 # number of empty sdrs between sequences

sequence_history = []
sdr_history = []
sequence_overlap = []

final_predicted_cells = []
final_active_cells = []
active_history = []
predicted_history= []

plot_state = False

for i in range(n_sequence_displays):
  print i
  
  # First just show some empty sdrs
  for j in range(steps_between_sequences):
    tm.compute(set([]), learn=True)

    if plot_state:
      figure(1)
      clf()
      subplot(311)
      plot([0, tm.numberOfColumns()], [0,0], 'k')
      xlim([0, tm.numberOfColumns()])
      title('Empty')
        
      subplot(3,1, (2,3))
      plot_tm_state(tm)
      pause(0.01)
    
    #sdr_history.append(0)
      
  # Now show one of the sequences
  which_seq = randint(num_sequences)
  sequence_history.append(which_seq)
  
  for j in range(num_sdr_per_seq):
    sdr_name = sdr_list[sequence_set[which_seq,j]]
    sdr = sdr_dict[sdr_name]
    predicted_cells = tm.predictiveCells
      
    tm.compute(set(find(sdr)), learn=True)
      
    if plot_state:
      figure(1)
      clf()
      subplot(311)
      plot(find(sdr), zeros(len(find(sdr))), '|k', ms=10)
      xlim([0, tm.numberOfColumns()])        
      title(sdr_name)
          
      subplot(3,1,(2,3))
      plot_tm_state(tm)
      pause(0.01)
      
    active_history.append(tm.activeCells)
    predicted_history.append(predicted_cells)
      
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


for i in [7]: #range(num_sequences):
  figure('sdr history - ' + str(i))
  clf()

  sequence_idxs = find(array(sequence_history) == i)
  
  num_display_trials = min(20, len(sequence_idxs))
  
  for j in range(num_display_trials):
    for k in range(num_sdr_per_seq):
      print i,j,k
      subplot(num_sdr_per_seq, num_display_trials, j + num_display_trials * k + 1)
          
      active_idx = sequence_idxs[j] *  num_sdr_per_seq + k           
        
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

