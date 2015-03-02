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

def plotActivePredicted(activeCells, predictedCells, cellsPerColumn):
  '''Visualization of tm state during one timestep
  
  @param activeCells set of cells in the active state
  @param predictedCells set of cells in the predicted state from previous step
  @param cellsPerColumn number of cells in a column
  
  '''
    
  activeUnpredicted = activeCells - predictedCells
  activePredicted = activeCells & predictedCells
  predictedUnactive = predictedCells - activeCells
  
  colIdx = zeros(len(activeUnpredicted))
  cellIdx = zeros(len(activeUnpredicted))
  for i, cell in enumerate(activeUnpredicted):
    colIdx[i] = cell // cellsPerColumn
    cellIdx[i] = cell % cellsPerColumn

  plot(colIdx, cellIdx, 'o', c='y')
  
  colIdx = zeros(len(predictedUnactive))
  cellIdx = zeros(len(predictedUnactive))
  for i, cell in enumerate(predictedUnactive):
    colIdx[i] = cell // cellsPerColumn
    cellIdx[i] = cell % cellsPerColumn
      
  plot(colIdx, cellIdx, 'o', c='c')
  colIdx = zeros(len(activePredicted))
  cellIdx = zeros(len(activePredicted))
  for i, cell in enumerate(activePredicted):
    colIdx[i] = cell // cellsPerColumn
    cellIdx[i] = cell % cellsPerColumn
      
  plot(colIdx, cellIdx, 'o', c='r')


def plotTMHistorySequence(filename, plotSequence=0, display=arange(20), outfile=None):
  '''plots the history of an individual sequence within a set of random sequences
  
  @param: filename filename of simulation data
  @param: plotSequence which sequence to plot, 0
  @param: display which trials to display, arange(20)
  @param: outfile the filename save figure. False does not save, None auto-generate name. None  
  
  @return: fh the handle to the figure
  @return: outfile the filename of the figure.
  '''
  
  with load(filename) as data:
    sequenceHistory = data['sequenceHistory']
    sdrList = data['sdrList']
    whichSequence = data['whichSequence']
    activeHistory = data['activeHistory']
    runID = data['runID']
    predictedHistory = data['predictedHistory']
    sequenceParams = data['sequenceParams'][0]
    cellsPerColumn = data['cellsPerColumn']
      
  fh = figure('sdr history - ' + str(plotSequence), figsize=(12,18))
  clf()

  sequenceStartIdxs = find(array(whichSequence) == plotSequence)
  
  seqDisplay = sequenceStartIdxs[display[display < len(sequenceStartIdxs)]]
  
  numDisplayTrials = len(seqDisplay)
  
  if numDisplayTrials == 0:
    print "Nothing to display!"
    return None,None
  
  for j, trial in enumerate(seqDisplay):
    
    subplot(numDisplayTrials, sequenceParams['sequenceLength'] + 1, j * (sequenceParams['sequenceLength']+1) + 1)
    activeCells = activeHistory[trial-1]
    predictedCells = predictedHistory[trial-1]
    
    plotActivePredicted(activeCells, predictedCells, cellsPerColumn)
    title(sdrList[trial-1])
    yticks([])
    xticks([])
    
    
    for k in range(sequenceParams['sequenceLength']):
      
      activeCells = activeHistory[trial+k]
      predictedCells = predictedHistory[trial+k]
      
      #subplot(num_display_trials, sequence_params['sequence_length'], j * sequence_params['sequence_length'] + k + 1)
      
      subplot(numDisplayTrials, sequenceParams['sequenceLength'] + 1, j * (sequenceParams['sequenceLength']+1) + k + 2)
      
      plotActivePredicted(activeCells, predictedCells, cellsPerColumn)
      
      xticks([])
      
      #if j == 0:
      title(sdrList[trial+k])
        
      if k == 0:
        yticks([])
        #yticks([cellsPerColumn/2], (str(trial), ))
      else:
        yticks([])
  
  if outfile is None:
    # Then make a filename
    outfile = ('history_sequence-s' + str(plotSequence) 
      + '-d%02d=%02d=%02d'% (display[0], mean(diff(display)), display[numDisplayTrials-1])
      + '-i' + '%05d' % runID
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')
    
  return fh, outfile

def plotTMHistoryAll(filename, display=arange(40), outfile=None):
  '''plots the history of all sequences and activity

  @param: filename filename of simulation data
  @param: display which trials to display, arange(40)
  @param: outfile the filename save figure. False does not save, None auto-generate name. None  
  
  @return: fh the handle to the figure
  @return: outfile the filename of the figure.
  '''  
  
  with load(filename) as data:
    sequenceHistory = data['sequenceHistory']
    sdrList = data['sdrList']
    activeHistory = data['activeHistory']
    predictedHistory = data['predictedHistory']
    runID = data['runID']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
       
  
  fh = figure('sdr history - ' + str(runID), figsize=(12,18))
  clf()
  
  display = display[display < len(sdrList)]
  
  numDisplayTrials = len(display)
  
  subplotDim = ceil(sqrt(numDisplayTrials))
  
  for j, trial in enumerate(display):
    subplot(subplotDim, subplotDim, j+1)
    
    plotActivePredicted(activeHistory[trial], predictedHistory[trial], cellsPerColumn)
  
    xticks([])
    yticks([cellsPerColumn/2], (str(trial), ))
    title(sdrList[trial])
  
  
  if outfile is None:
    # Then make a filename
    outfile = ('history_all-d'+ str(numDisplayTrials) 
      + '-i' + '%05d' % runID
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')
  
  return fh, outfile
  
def plotTMOverlapAll(filename, outfile=None, smoothFactor=10):
  '''plots prediction overlap for all displays
  
  
  @param: filename filename of simulation data
  @param: outfile the filename save figure. False does not save, None auto-generate name. None  
  @param: smoothFactor size of the running average, 10
  
  @return: fh the handle to the figure
  @return: outfile the filename of the figure.  
  '''
  
  with load(filename) as data:
    sequenceHistory = data['sequenceHistory']
    sdrList = data['sdrList']
    activeHistory = data['activeHistory']
    predictedHistory = data['predictedHistory']
    runID = data['runID']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
    sequenceOverlap = data['sequenceOverlap']
  
  fh = figure('Sequence Prediction Overlap')
  clf()
    
  plot(sequenceOverlap, ':k', lw=1)
  plot(convolve(sequenceOverlap, ones(smoothFactor)/smoothFactor, mode='same'), 'k', lw=3)
  
  xlabel('Time')
  ylabel('Error')

  if outfile is None:
    # Then make a filename
    outfile = ('overlap_all-l'+ str(len(sequenceOverlap)) 
      + '-i' + '%05d' % runID
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')
    
  return fh, outfile

def plotTMErrorAll(filename, outfile=None, smoothFactor=10):
  '''plots prediction error for all displays

  @param: filename filename of simulation data
  @param: outfile the filename save figure. False does not save, None auto-generate name. None  
  @param: smoothFactor size of the running average, 10
  
  @return: fh the handle to the figure
  @return: outfile the filename of the figure.  
  '''
  
  with load(filename) as data:
    sequenceHistory = data['sequenceHistory']
    sdrList = data['sdrList']
    activeHistory = data['activeHistory']
    predictedHistory = data['predictedHistory']
    runID = data['runID']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
    sequenceOverlap = data['sequenceOverlap']
    sequenceError = data['sequenceError']
    sequenceParams = data['sequenceParams'][0]
  
  fh = figure('Prediction Error')
  clf()
  
  
  plot(sequenceError, ':k', lw=1)
  plot(convolve(sequenceError, ones(smoothFactor)/smoothFactor, mode='same'), 'k', lw=3)
  
  # This is the theoretical minimum (somewhat probabilistic)
  bestError = ((sequenceParams['sequenceLength'] - 1) * (1 - sequenceParams['pRandSDR'])) 
    
  if sequenceParams['forceRandSDR']:
    # Then there is one more rand sdr per sequence
    bestError /= (sequenceParams['sequenceLength'] + 1)
  else:
    bestError /= sequenceParams['sequenceLength']

  
  plot([0, len(sequenceError)], [bestError, bestError], 'r')  
  
  xlabel('Time')
  ylabel('Error')
  
  
  if outfile is None:
    # Then make a filename
    outfile = ('error_all-l'+ str(len(sequenceError)) 
      + '-i' + '%05d' % runID
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')
  
  return fh, outfile
  
def plotTMErrorSequence(filename, plotSequences=None, outfile=None):
  '''plots prediction error for all predictable elements in a given set of sequences
  
  @param: filename filename of simulation data
  @param: plotSequence which sequence to plot, 0
  @param: outfile the filename save figure. False does not save, None auto-generate name. None  
  
  @return: fh the handle to the figure
  @return: outfile the filename of the figure.
    
  '''
  
  with load(filename) as data:
    sequenceHistory = data['sequenceHistory']
    sdrList = data['sdrList']
    activeHistory = data['activeHistory']
    predictedHistory = data['predictedHistory']
    runID = data['runID']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
    sequenceOverlap = data['sequenceOverlap']
    sequenceError = data['sequenceError']
    sequenceParams = data['sequenceParams'][0]
    whichSequence = data['whichSequence']
  

  if plotSequences is None:
    # Then plot all the sequences
    plotSequences = arange(sequenceParams['numSequences'])
  
  fh = figure('Sequence Prediction Error')
  clf()
  
  colors = get_cmap('Set3', len(plotSequences))

  maxSequenceDisplays = 0
  
  seqError = []  
  
  for i, seq in enumerate(plotSequences):
    sequenceStartIdxs = find(array(whichSequence) == seq)
    seqError.insert(i, [])    

    if len(sequenceStartIdxs) * (sequenceParams['sequenceLength'] - 1) > maxSequenceDisplays:
      maxSequenceDisplays = len(sequenceStartIdxs) * (sequenceParams['sequenceLength'] - 1)
    
    for j, seqStart in enumerate(sequenceStartIdxs):
      for k in range(1, sequenceParams['sequenceLength']):
        idx = seqStart + k
        if idx < len(sequenceError):
          seqError[i].append(sequenceError[idx])

  # put everything into array to average together  
  errorExposures = nan * zeros((len(plotSequences), maxSequenceDisplays))
  for i, se in enumerate(seqError):
    errorExposures[i, :len(se)] = se
  
    plot(se, lw=3, label='sequence ' + str(plotSequences[i]), c=colors(i))
    
  plot(nanmean(errorExposures, 0), 'k', lw=4)
   
  xlabel('Time')
  ylabel('Error')
  
  
  if outfile is None:
    # Then make a filename
    outfile = ('error_seq-s' + str(len(plotSequences)) + '-l'+ str(len(sequenceError)) 
      + '-i' + '%05d' % runID
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')
    
  return fh, outfile
  

def plotTMErrorSequenceEnd(filename, plotSequences=None, outfile=None):
  '''plots prediction error for last element in a given set of sequences
  
  @param: filename filename of simulation data
  @param: plotSequence which sequence to plot, 0
  @param: outfile the filename save figure. False does not save, None auto-generate name. None  
  
  @return: fh the handle to the figure
  @return: outfile the filename of the figure.
  '''
  
  with load(filename) as data:
    sequenceHistory = data['sequenceHistory']
    sdrList = data['sdrList']
    activeHistory = data['activeHistory']
    predictedHistory = data['predictedHistory']
    runID = data['runID']
    numberOfColumns = data['numberOfColumns']
    cellsPerColumn = data['cellsPerColumn']
    sequenceOverlap = data['sequenceOverlap']
    sequenceError = data['sequenceError']
    sequenceParams = data['sequenceParams'][0]
    whichSequence = data['whichSequence']
  

  if plotSequences is None:
    # Then plot all the sequences
    plotSequences = arange(sequenceParams['numSequences'])
  
  fh = figure('Sequence Prediction Error End')
  clf()
  
  colors = get_cmap('Set3', len(plotSequences))

  maxSequenceDisplays = 0
  
  seqError = []  
  
  for i, seq in enumerate(plotSequences):
    sequenceStartIdxs = find(array(whichSequence) == seq)
    seqError.insert(i, [])    

    if len(sequenceStartIdxs) > maxSequenceDisplays:
      maxSequenceDisplays = len(sequenceStartIdxs)
      
    for j, seqStart in enumerate(sequenceStartIdxs):
      idx = seqStart + sequenceParams['sequenceLength'] - 1
      
      if idx < len(sequenceError):
        # In case the last part of the sequence was never shown
        seqError[i].insert(j, sequenceError[idx])

  # put everything into array to average together  
  errorExposures = nan * zeros((len(plotSequences), maxSequenceDisplays))
  for i, se in enumerate(seqError):
    errorExposures[i, :len(se)] = se
  
    plot(se, lw=3, label='sequence ' + str(plotSequences[i]), c=colors(i))
    
  plot(nanmean(errorExposures, 0), 'k', lw=4)
   
  xlabel('Time')
  ylabel('Error')
  
  if len(plotSequences) < 10:
    legend()
    
  if outfile is None:
    # Then make a filename
    outfile = ('error_seq_end-s' + str(len(plotSequences)) + '-l'+ str(len(sequenceError)) 
      + '-i' + '%05d' % runID
      + '-n' + datetime.date.today().strftime('%y%m%d'))
  
  if outfile:
    fh.savefig(outfile + '.png', format='png')
    #fh.savefig(outfile, format='eps')

  return fh, outfile
  

#%%



def createSDRList(params):
  '''Creates a list of sdrs to show a temporal memory with predictable sequences
  
  @param: params dictionary of parameters with the following key values:
    totalDisplays number of SDRs to return, 100
    numElements number of random SDRs to choose from, 100
    numSequences number of predictable sequences, 4
    sequenceLength number of SDRs in sequence, 2
    sequenceOrder order of the sequences (number of shared elements), 0
    pRandSDR probability of random SDR appearing between sequences, 0.5
    forceRandSDR force a random SDR between sequences, False
    uniqueRandSDR random SDRs and sequences completely unique, False
  
  @return: sdrList the index of each sdr
  @return: which_sequence the beginnings of each unique sequence, -1 means random sdr
  @return: sequences each individual sequence
  @return: params the parameters for the sequence creation
  '''
  
  defaultParams = {}
  defaultParams['totalDisplays'] = 100
  defaultParams['numElements'] = 100
  defaultParams['numSequences'] = 4
  defaultParams['sequenceLength'] = 2
  defaultParams['sequenceOrder'] = 0
  defaultParams['pRandSDR'] = 0.5
  defaultParams['forceRandSDR'] = False
  defaultParams['uniqueRandSDR'] = False
  
  for key in defaultParams.keys():
    if key not in params.keys():
      params[key] = defaultParams[key]
  
  if params['sequenceLength'] <= params['sequenceOrder']:
    print "sequenceLength must be larger than sequence order"
    return
    
  if params['numElements'] < params['numSequences'] * params['sequenceLength']:
    print "too many sequences for uniqueness"
    return
  
  # now make unique sequences
  sequencesPerm = permutation(arange(params['numElements']))
  sequences = reshape(sequencesPerm[:params['numSequences'] * params['sequenceLength']],
                      (params['numSequences'], params['sequenceLength']))
                      
  leftOverSDRs = sequencesPerm[params['numSequences']*params['sequenceLength']:]

  # now insert the high order elements
  if params['sequenceOrder'] > 0:    
    # make the higher-order sequence subset, and replace the middle
    hoSeq = randint(params['numElements'], size=params['sequenceOrder'])  
    sequences[:, 1:(params['sequenceOrder']+1)] = tile(hoSeq, (params['numSequences'], 1))
  
  # need the extra numElements to make sure we don't index out of bounds
  sdrList = zeros(params['totalDisplays'] + params['numElements']) 
  whichSequence = -1 * ones(params['totalDisplays'] + params['numElements'])

  c=0  
  while c < params['totalDisplays']:
    
    if rand() < params['pRandSDR']:
      # then pick a random element
      if params['uniqueRandSDR']:
        # Then pick random sdrs excluded from sequence sdrs
        sdrList[c] = leftOverSDRs[randint(len(leftOverSDRs))] 
      else:
        sdrList[c] = randint(params['numElements'])
      c += 1
    else:
      # Then pick a sequence
      seqIdx = randint(params['numSequences'])
      whichSequence[c] = seqIdx
      
      for j in range(params['sequenceLength']):
        sdrList[c] = sequences[seqIdx, j]        
        c += 1
      
      if params['forceRandSDR']:
        sdrList[c] = randint(params['numElements'])
        c += 1
      
  sdrList = sdrList[:params['totalDisplays']]
  whichSequence = whichSequence[:params['totalDisplays']]
  
  return (sdrList, whichSequence, sequences, params)


def runTMSequenceExperiment(tm, sequenceParams, filename=None, runID=None):
  '''Runs a temporal memory sequence learning experiment

  @param: tm a temporal memory instance
  @param: sequenceParams dictionary of sequence parameters
  @param: filename filename to save simulation data, None generates filename, None
  @param: runID identifier for the simulation, None generates runID, None

  @return: filename the name of the file for the simulation data.  
  '''
  
  (sdrList, whichSequence, sequences, sequenceParams) = createSDRList(sequenceParams)  
  
  if runID is None:
     runID = randint(1e6)
  
  if filename is None:
    # Generate a filename
    filename = ('tm_sequence_experiment-l' + str(len(sdrList)) 
      + '-i' + '%05d' % runID
      + '-n' + datetime.date.today().strftime('%y%m%d') + '.npz') 
      
  
  sequenceOverlap = []
  sequenceError = []
  activeHistory = []
  predictedHistory= []
  sequenceHistory = [] # the true sdrs, these have specific noise
  
  plotState = False
  
  pFlip10 = 0 # probability 1 flips to 0
  pFlip01 = 0 # probability 0 flips to 1

  sdrEncoder = SDRCategoryEncoder(tm.numberOfColumns(), 40)
  sdrDict = {}

  for sdr in unique(sdrList):
    sdrDict[sdr] = sdrEncoder.encode(sdr)
  
  for i,sdrName in enumerate(sdrList):
    print i
    sdr = sdrDict[sdrName]
    predictedCells = tm.predictiveCells
    
    sdrOnIdxs = find(sdr)
    sdrOffIdxs = find(~sdr)
    flip10 = rand(len(sdrOnIdxs)) < pFlip10
    flip01 = rand(len(sdrOffIdxs)) < pFlip01
    
    sdrAddIdxs = sdrOffIdxs[flip01]
    sdrRemoveIdxs = sdrOnIdxs[flip10]
    
    sdrOnIdxs = set(sdrOnIdxs)
    sdrOnIdxs = sdrOnIdxs - set(sdrRemoveIdxs)
    sdrOnIdxs = sdrOnIdxs | set(sdrAddIdxs)
    
    tm.compute(sdrOnIdxs, learn=True)
      
    if plotState:
      figure(1)
      clf()
      subplot(311)
      plot(find(sdr), zeros(len(find(sdr))), '|k', ms=10)
      plot(sdrAddIdxs, zeros(len(sdrAddIdxs)), '|b', ms=10)
      plot(sdrRemoveIdxs, zeros(len(sdrRemoveIdxs)), '.r', ms=10)
      xlim([0, tm.numberOfColumns()])        
      title(sdrName)
          
      subplot(3,1,(2,3))
      plotActivePredicted(tm.activeCells, predictedCells, tm.cellsPerColumn)
      pause(0.01)
      
    activeHistory.append(tm.activeCells)
    predictedHistory.append(predictedCells)
    sequenceHistory.append(sdrOnIdxs)
    
    if len(predictedCells) == 0:
      # Then nothing was predicted so 100% error
      sequenceError.append(1)
    else:  
      sequenceError.append(1 - len(tm.activeCells & predictedCells) / len(predictedCells))
    
    predictedColumns = set()
    for cell in predictedCells:
        predictedColumns.add(tm.columnForCell(cell))
    activeColumns = set()
    for cell in tm.activeCells:
        activeColumns.add(tm.columnForCell(cell))
    
    sequenceOverlap.append(len(predictedColumns & activeColumns))
    
  # Instead of returning arguments, just save everything to a file    
    
  savez_compressed(filename, activeHistory=activeHistory, predictedHistory=predictedHistory, 
                   sequenceOverlap=sequenceOverlap, sequenceError=sequenceError,
                   sdrList=sdrList, sequenceHistory=sequenceHistory,
                   whichSequence=whichSequence, sequences=sequences, 
                   sequenceParams=[sequenceParams], # Need to make list to save dict correctly 
                   pFlip01=pFlip01, pFlip10=pFlip10, runID=runID,
                   cellsPerColumn=tm.cellsPerColumn, numberOfColumns=tm.numberOfColumns())

  return filename


#%% Run some experiments

if __name__ == '__main__':
  #%%  
  close('all')
  
  
  tm = TemporalMemory(minThreshold=30, activationThreshold=30,  maxNewSynapseCount=40)
  
  sequenceParams = {}
  sequenceParams['totalDisplays'] = 200
  sequenceParams['numElements'] = 1000
  sequenceParams['numSequences'] = 8
  sequenceParams['sequenceLength'] = 4
  sequenceParams['sequenceOrder'] = 0
  sequenceParams['pRandSDR'] = 0
  sequenceParams['forceRandSDR'] = True
  sequenceParams['uniqueRandSDR'] = False
   
  # Just comment this out to re-run and re-plot without running the simulation 
  filename = runTMSequenceExperiment(tm, sequenceParams)
  
  #%%
  plotTMHistoryAll(filename, arange(400))
  #%%
  plotTMHistorySequence(filename, 6, arange(0, 30))
  #plot_tm_history_sequence(filename, 0, arange(100, 130))
  
  #%%
  plotTMOverlapAll(filename)
  plotTMErrorAll(filename)
  plotTMErrorSequence(filename)
  plotTMErrorSequenceEnd(filename)
  
  #%%
  
  with load(filename) as data:
    sequences = data['sequences']
    sdrList = data['sdrList']
    whichSequence = data['whichSequence']
    
  seqIdx = find(sdrList == 25)
  
  print sdrList[seqIdx + 1]
