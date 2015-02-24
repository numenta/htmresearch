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
This module contains plotting tools and functions for running sequence 
experiments.
"""

# This overrides the floor integer division
from __future__ import division

from pylab import *

from nupic.research.temporal_memory import TemporalMemory
from nupic.encoders.sdrcategory import SDRCategoryEncoder

import datetime

#%%
close('all')

#%%

def plot_active_predicted(active_cells, predicted_cells, cellsPerColumn):
  '''plots the predicted and active cells'''
    
  active_unpredicted = active_cells - predicted_cells
  active_predicted = active_cells & predicted_cells
  predicted_unactive = predicted_cells - active_cells
  
  col_idx = zeros(len(active_unpredicted))
  cell_idx = zeros(len(active_unpredicted))
  for i, cell in enumerate(active_unpredicted):
    col_idx[i] = cell // cellsPerColumn
    cell_idx[i] = cell % cellsPerColumn

  plot(col_idx, cell_idx, 'o', c='y')
  
  col_idx = zeros(len(predicted_unactive))
  cell_idx = zeros(len(predicted_unactive))
  for i, cell in enumerate(predicted_unactive):
    col_idx[i] = cell // cellsPerColumn
    cell_idx[i] = cell % cellsPerColumn
      
  plot(col_idx, cell_idx, 'o', c='c')
  col_idx = zeros(len(active_predicted))
  cell_idx = zeros(len(active_predicted))
  for i, cell in enumerate(active_predicted):
    col_idx[i] = cell // cellsPerColumn
    cell_idx[i] = cell % cellsPerColumn
      
  plot(col_idx, cell_idx, 'o', c='r')


def plot_tm_history_sequence(filename, plot_sequence=0, display=arange(20), outfile=None):
  '''plots the history of an individual sequence within a set of random sequences'''
  
  with load(filename) as data:
    sequence_history = data['sequence_history']
    sequence_list = data['sequence_list']
    which_sequence = data['which_sequence']
    active_history = data['active_history']
    run_id = data['run_id']
    predicted_history = data['predicted_history']
    sequence_params = data['sequence_params'][0]
    cellsPerColumn = data['cellsPerColumn']
      
  fh = figure('sdr history - ' + str(plot_sequence), figsize=(12,18))
  clf()

  sequence_start_idxs = find(array(which_sequence) == plot_sequence)
  
  seq_display = sequence_start_idxs[display[display < len(sequence_start_idxs)]]
  
  num_display_trials = len(seq_display)
  
  if num_display_trials == 0:
    print "Nothing to display!"
    return
  
  for j, trial in enumerate(seq_display):
    
    subplot(num_display_trials, sequence_params['sequence_length'] + 1, j * (sequence_params['sequence_length']+1) + 1)
    active_cells = active_history[trial-1]
    predicted_cells = predicted_history[trial-1]
    
    plot_active_predicted(active_cells, predicted_cells, cellsPerColumn)
    title(sequence_list[trial-1])
    
    for k in range(sequence_params['sequence_length']):
      
      active_cells = active_history[trial+k]
      predicted_cells = predicted_history[trial+k]
      
      #subplot(num_display_trials, sequence_params['sequence_length'], j * sequence_params['sequence_length'] + k + 1)
      
      subplot(num_display_trials, sequence_params['sequence_length'] + 1, j * (sequence_params['sequence_length']+1) + k + 2)
      
      plot_active_predicted(active_cells, predicted_cells, cellsPerColumn)
      
      xticks([])
      
      #if j == 0:
      title(sequence_list[trial+k])
        
      if k == 0:
        yticks([cellsPerColumn/2], (str(trial), ))
      else:
        yticks([])
  
  if outfile is None:
    # Then make a filename
    outfile = ('history_sequence-s' + str(plot_sequence) 
      + '-d%02d=%02d=%02d'% (display[0], mean(diff(display)), display[num_display_trials-1])
      + '-i' + '%05d' % run_id
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')

def plot_tm_history_all(filename, display=arange(40), outfile=None):
  '''plots the history of all sequences and activity'''  
  
  with load(filename) as data:
    sequence_history = data['sequence_history']
    sequence_list = data['sequence_list']
    active_history = data['active_history']
    predicted_history = data['predicted_history']
    run_id = data['run_id']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
       
  
  fh = figure('sdr history - ' + str(run_id), figsize=(12,18))
  clf()
  
  display = display[display < len(sequence_list)]
  
  num_display_trials = len(display)
  
  subplot_dim = ceil(sqrt(num_display_trials))
  
  for j, trial in enumerate(display):
    subplot(subplot_dim, subplot_dim, j+1)
    
    plot_active_predicted(active_history[trial], predicted_history[trial], cellsPerColumn)
  
    xticks([])
    yticks([cellsPerColumn/2], (str(trial), ))
    title(sequence_list[trial])
  
  
  if outfile is None:
    # Then make a filename
    outfile = ('history_all-d'+ str(num_display_trials) 
      + '-i' + '%05d' % run_id
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')

  
def plot_tm_overlap_all(filename, outfile=None):
  '''plots prediction overlap for all displays'''
  
  with load(filename) as data:
    sequence_history = data['sequence_history']
    sequence_list = data['sequence_list']
    active_history = data['active_history']
    predicted_history = data['predicted_history']
    run_id = data['run_id']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
    sequence_overlap = data['sequence_overlap']
  
  fh = figure('Sequence Prediction Overlap')
  clf()
  
  smooth_factor = 10
  
  plot(sequence_overlap, ':k', lw=1)
  plot(convolve(sequence_overlap, ones(smooth_factor)/smooth_factor, mode='same'), 'k', lw=3)
  
  xlabel('Time')
  ylabel('Error')

  if outfile is None:
    # Then make a filename
    outfile = ('overlap_all-l'+ str(len(sequence_overlap)) 
      + '-i' + '%05d' % run_id
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')

def plot_tm_error_all(filename, outfile=None):
  '''plots prediction error for all displays'''
  
  with load(filename) as data:
    sequence_history = data['sequence_history']
    sequence_list = data['sequence_list']
    active_history = data['active_history']
    predicted_history = data['predicted_history']
    run_id = data['run_id']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
    sequence_overlap = data['sequence_overlap']
    sequence_error = data['sequence_error']
    sequence_params = data['sequence_params'][0]
  
  fh = figure('Prediction Error')
  clf()
  
  smooth_factor = 10  
  
  plot(sequence_error, ':k', lw=1)
  plot(convolve(sequence_error, ones(smooth_factor)/smooth_factor, mode='same'), 'k', lw=3)
  
  # This is the theoretical minimum (somewhat probabilistic)
  best_error = ((sequence_params['sequence_length'] - 1) * (1 - sequence_params['p_rand_sdr']) 
    / sequence_params['sequence_length'])
  
  plot([0, len(sequence_error)], [best_error, best_error], 'r')  
  
  xlabel('Time')
  ylabel('Error')
  
  
  if outfile is None:
    # Then make a filename
    outfile = ('error_all-l'+ str(len(sequence_error)) 
      + '-i' + '%05d' % run_id
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')

  
def plot_tm_error_sequence(filename, plot_sequences=None, outfile=None):
  '''plots prediction error for all predictable elements in a given set of sequences'''
  
  with load(filename) as data:
    sequence_history = data['sequence_history']
    sequence_list = data['sequence_list']
    active_history = data['active_history']
    predicted_history = data['predicted_history']
    run_id = data['run_id']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
    sequence_overlap = data['sequence_overlap']
    sequence_error = data['sequence_error']
    sequence_params = data['sequence_params'][0]
    which_sequence = data['which_sequence']
  

  if plot_sequences is None:
    # Then plot all the sequences
    plot_sequences = arange(sequence_params['num_sequences'])
  
  fh = figure('Sequence Prediction Error')
  clf()
  
  colors = get_cmap('Set3', len(plot_sequences))

  max_sequence_displays = 0
  
  seq_error = []  
  
  for i, seq in enumerate(plot_sequences):
    sequence_start_idxs = find(array(which_sequence) == seq)
    seq_error.insert(i, [])    

    if len(sequence_start_idxs) * (sequence_params['sequence_length'] - 1) > max_sequence_displays:
      max_sequence_displays = len(sequence_start_idxs) * (sequence_params['sequence_length'] - 1)
    
    for j, seq_start in enumerate(sequence_start_idxs):
      for k in range(1, sequence_params['sequence_length']):
        idx = seq_start + k
        if idx < len(sequence_error):
          seq_error[i].append(sequence_error[idx])

  # put everything into array to average together  
  error_exposures = nan * zeros((len(plot_sequences), max_sequence_displays))
  for i, se in enumerate(seq_error):
    error_exposures[i, :len(se)] = se
  
    plot(se, lw=3, label='sequence ' + str(plot_sequences[i]), c=colors(i))
    
  plot(nanmean(error_exposures, 0), 'k', lw=4)
   
  xlabel('Time')
  ylabel('Error')
  
  
  if outfile is None:
    # Then make a filename
    outfile = ('error_seq-s' + str(len(plot_sequences)) + '-l'+ str(len(sequence_error)) 
      + '-i' + '%05d' % run_id
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')
  

def plot_tm_error_sequence_end(filename, plot_sequences=None, outfile=None):
  '''plots prediction error for last element in a given set of sequences'''
  
  with load(filename) as data:
    sequence_history = data['sequence_history']
    sequence_list = data['sequence_list']
    active_history = data['active_history']
    predicted_history = data['predicted_history']
    run_id = data['run_id']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
    sequence_overlap = data['sequence_overlap']
    sequence_error = data['sequence_error']
    sequence_params = data['sequence_params'][0]
    which_sequence = data['which_sequence']
  

  if plot_sequences is None:
    # Then plot all the sequences
    plot_sequences = arange(sequence_params['num_sequences'])
  
  fh = figure('Sequence Prediction Error End')
  clf()
  
  colors = get_cmap('Set3', len(plot_sequences))

  max_sequence_displays = 0
  
  seq_error = []  
  
  for i, seq in enumerate(plot_sequences):
    sequence_start_idxs = find(array(which_sequence) == seq)
    seq_error.insert(i, [])    

    if len(sequence_start_idxs) > max_sequence_displays:
      max_sequence_displays = len(sequence_start_idxs)
    
    for j, seq_start in enumerate(sequence_start_idxs):
      idx = seq_start + sequence_params['sequence_length'] - 1
      
      if idx < len(sequence_error):
        # In case the last part of the sequence was never shown
        seq_error[i].insert(j, sequence_error[idx])

  # put everything into array to average together  
  error_exposures = nan * zeros((len(plot_sequences), max_sequence_displays))
  for i, se in enumerate(seq_error):
    error_exposures[i, :len(se)] = se
  
    plot(se, lw=3, label='sequence ' + str(plot_sequences[i]), c=colors(i))
    
  plot(nanmean(error_exposures, 0), 'k', lw=4)
   
  xlabel('Time')
  ylabel('Error')
  
  if len(plot_sequences) < 10:
    legend()
    
  if outfile is None:
    # Then make a filename
    outfile = ('error_seq_end-s' + str(len(plot_sequences)) + '-l'+ str(len(sequence_error)) 
      + '-i' + '%05d' % run_id
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')

  

#%%



def createSequenceList(params):
  '''Creates a list of sequences to show a temporal memory
  
  @param: total_displays number of sequences to return
  @param: num_elements total number of random sdrs to use
  @param: num_sequences the number of predictable sequences
  @param: sequence_length number of sdrs in sequence
  @param: sequence_order order of common sequence, must be less than sequence_length
  @param: p_rand_sdr probability of a random sdr appearing between sequences
  
  @return: sequence_list the index of each sdr
  @return: which_sequence the beginnings of each unique sequence, -1 means random sdr
  @return: sequences each individual sequence
  @return: params the parameters for the sequence creation
  '''
  
  default_params = {}
  default_params['total_displays'] = 100
  default_params['num_elements'] = 100
  default_params['num_sequences'] = 4
  default_params['sequence_length'] = 2
  default_params['sequence_order'] = 0
  default_params['p_rand_sdr'] = 0.5
  
  
  for key in default_params.keys():
    if key not in params.keys():
      params[key] = default_params[key]
  
  if params['sequence_length'] <= params['sequence_order']:
    print "sequence_length must be larger than sequence order"
    return
    
  if params['num_elements'] < params['num_sequences'] * params['sequence_length']:
    print "too many sequences for uniqueness"
    return
  
  # now make unique sequences
  sequences_perm = permutation(arange(params['num_elements']))
  sequences = reshape(sequences_perm[:params['num_sequences'] * params['sequence_length']],
                      (params['num_sequences'], params['sequence_length']))

  # now insert the high order elements
  if params['sequence_order'] > 0:    
    # make the higher-order sequence subset, and replace the middle
    ho_seq = randint(params['num_elements'], size=params['sequence_order'])  
    sequences[:, 1:(params['sequence_order']+1)] = tile(ho_seq, (params['num_sequences'], 1))
  
  # need the extra num_elements to make sure we don't index out of bounds
  sequence_list = zeros(params['total_displays'] + params['num_elements']) 
  which_sequence = -1 * ones(params['total_displays'] + params['num_elements'])

  c=0  
  while c < params['total_displays']:
    
    if rand() < params['p_rand_sdr']:
      # then pick a random element
      sequence_list[c] = randint(params['num_elements'])
      c += 1
    else:
      # Then pick a sequence
      seq_idx = randint(params['num_sequences'])
      which_sequence[c] = seq_idx
      
      for j in range(params['sequence_length']):
        sequence_list[c] = sequences[seq_idx, j]
        
        c += 1
      
      # Uncomment to force random sdr between all sequences
      #sequence_list[c] = randint(params['num_elements'])
      #c += 1
      
  sequence_list = sequence_list[:params['total_displays']]
  which_sequence = which_sequence[:params['total_displays']]
  
  return (sequence_list, which_sequence, sequences, params)


def runTMSequenceExperiment(tm, sequence_params, filename=None, run_id=None):
  '''Runs a temporal memory sequence learning experiment'''
  
  (sequence_list, which_sequence, sequences, sequence_params) = createSequenceList(sequence_params)  
  
  if run_id is None:
     run_id = randint(1e6)
  
  if filename is None:
    # Generate a filename
    filename = ('tm_sequence_experiment-l' + str(len(sequence_list)) 
      + '-i' + '%05d' % run_id
      + '-n' + datetime.date.today().strftime('%y%m%d') + '.npz') 
      
  
  sequence_overlap = []
  sequence_error = []
  active_history = []
  predicted_history= []
  sequence_history = [] # the true sdrs, these have specific noise
  
  plot_state = True
  
  p_flip_10 = 0 # probability 1 flips to 0
  p_flip_01 = 0 # probability 0 flips to 1

  sdr_encoder = SDRCategoryEncoder(tm.numberOfColumns(), 40)
  sdr_dict = {}

  for sdr in unique(sequence_list):
    sdr_dict[sdr] = sdr_encoder.encode(sdr)
  
  for i,sdr_name in enumerate(sequence_list):
    print i
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
    
    tm.compute(sdr_on_idxs, learn=True)
      
    if plot_state:
      figure(1)
      clf()
      subplot(311)
      plot(find(sdr), zeros(len(find(sdr))), '|k', ms=10)
      plot(sdr_add_idxs, zeros(len(sdr_add_idxs)), '|b', ms=10)
      plot(sdr_remove_idxs, zeros(len(sdr_remove_idxs)), '.r', ms=10)
      xlim([0, tm.numberOfColumns()])        
      title(sdr_name)
          
      subplot(3,1,(2,3))
      plot_tm_state(tm)
      pause(0.01)
      
    active_history.append(tm.activeCells)
    predicted_history.append(predicted_cells)
    sequence_history.append(sdr_on_idxs)
    
    if len(predicted_cells) == 0:
      # Then nothing was predicted so 100% error
      sequence_error.append(1)
    else:  
      sequence_error.append(1 - len(tm.activeCells & predicted_cells) / len(predicted_cells))
    
    predicted_columns = set()
    for cell in predicted_cells:
        predicted_columns.add(tm.columnForCell(cell))
    active_columns = set()
    for cell in tm.activeCells:
        active_columns.add(tm.columnForCell(cell))
    
    sequence_overlap.append(len(predicted_columns & active_columns))
    
  # Instead of returning arguments, just save everything to a file    
    
  savez_compressed(filename, active_history=active_history, predicted_history=predicted_history, 
                   sequence_overlap=sequence_overlap, sequence_error=sequence_error,
                   sequence_list=sequence_list, sequence_history=sequence_history,
                   which_sequence=which_sequence, sequences=sequences, 
                   sequence_params=[sequence_params], # Need to make list to save dict correctly 
                   p_flip_01=p_flip_01, p_flip_10=p_flip_10, run_id=run_id,
                   cellsPerColumn=tm.cellsPerColumn, numberOfColumns=tm.numberOfColumns())

  return filename


#%% Run some experiments

tm = TemporalMemory(minThreshold=40, activationThreshold=40,  maxNewSynapseCount=40)

sequence_params = {}
sequence_params['total_displays'] = 1000
sequence_params['num_elements'] = 100
sequence_params['num_sequences'] = 8
sequence_params['sequence_length'] = 4
sequence_params['sequence_order'] = 0
sequence_params['p_rand_sdr'] = 0
 
# Just comment this out to re-run and re-plot without running the simulation 
#filename = runTMSequenceExperiment(tm, sequence_params)

#%%
plot_tm_history_all(filename, arange(400))
#%%
plot_tm_history_sequence(filename, 5, arange(10, 30))
#plot_tm_history_sequence(filename, 0, arange(100, 130))

#%%
plot_tm_overlap_all(filename)
plot_tm_error_all(filename)
plot_tm_error_sequence(filename)
plot_tm_error_sequence_end(filename)
