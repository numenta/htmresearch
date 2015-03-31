# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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

import numpy
import csv
from optparse import OptionParser
import sys

from sensorimotor.orphan_temporal_memory import OrphanTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)

class MonitoredTemporalMemory(TemporalMemoryMonitorMixin, OrphanTemporalMemory): pass


def letterSequence(letters, w=40):
  """
  Return a list of input vectors corresponding to sequence of letters.
  The vector for each letter has w contiguous bits ON and represented as a
  sequence of non-zero indices.
  """
  sequence = []
  for letter in letters:
    i = ord(letter) - ord('A')
    sequence.append(set(range(i*w,(i+1)*w)))
  return sequence


def getRandomVector(w=40, n=2048):
  "Return a list of w random indices out of a vector of n elements"
  return set(numpy.random.permutation(n)[0:w])


def getNextSequenceChunk(it, w=40, n=2048):
  """
  Given an iteration index, returns a list of vectors to be appended to the
  input stream, as well as a string label identifying the sequence
  """
  if it%20==3:
    vecs = letterSequence("ABCDEF")
    label="ABCDEF"
  elif it%20==13:
    vecs = letterSequence("GBIJKL")
    label="GBIJKL"
  else:
    vecs= [getRandomVector(w, n)]
    label="."

  return vecs,label


def getLowOrderSequenceChunk(it, w=40, n=2048):
  """
  Given an iteration index, returns a list of vectors to be appended to the
  input stream, as well as a string label identifying the sequence
  """
  if it%10==3:
    s = numpy.random.randint(3)
    if s==0:
      label="ABCDEF"
      vecs = letterSequence(label)
    elif s==1:
      label="CBEAFD"
      vecs = letterSequence(label)
    else:
      label="DABFEC"
      vecs = letterSequence(label)
  else:
    vecs= [getRandomVector(w, n)]
    label="."

  return vecs,label


def getHighOrderSequenceChunk(it, switchover = 1000, w=40, n=2048):
  """
  Given an iteration index, returns a list of vectors to be appended to the
  input stream, as well as a string label identifying the sequence. This
  version generates a bunch of high order sequences. The first element always
  provides sufficient context to predict the rest of the elements.

  After switchover iterations, it will generate a different set of sequences.
  """
  if it%10==3:
    s = numpy.random.randint(5)
    if it <= switchover:
      if s==0:
        label="XABCDE"
      elif s==1:
        label="YCBEAF"
      elif s==2:
        label="GHIJKL"
      elif s==3:
        label="WABCMN"
      else:
        label="ZDABCE"
    else:
      if s==0:
        label="XCBEAF"
      elif s==1:
        label="YABCDE"
      elif s==2:
        label="GABCMN"
      elif s==3:
        label="WHIJKL"
      else:
        label="ZDHICF"

    vecs = letterSequence(label)
  else:
    vecs= [getRandomVector(w, n)]
    label="."

  return vecs,label


def printSegment(tm, segment, connections):
  cell = connections.cellForSegment(segment)
  synapses = connections.synapsesForSegment(segment)
  print "segment id=",segment
  print "   cell=",cell
  print "   col =",tm.columnForCell(cell)
  print "   synapses=",
  for synapse in synapses:
    synapseData = connections.dataForSynapse(synapse)
    permanence = synapseData.permanence
    presynapticCell = synapseData.presynapticCell
    print "%d:%g" % (presynapticCell,permanence),
  print


def computePredictionAccuracy(pac, pic):
  """
  Given a temporal memory instance return the prediction accuracy. The accuracy
  is computed as 1 - (#correctly predicted cols / # predicted cols). The
  accuracy is 0 if there were no predicted columns.
  """
  pcols = float(pac + pic)
  if pcols == 0:
    return 0.0
  else:
    return (pac / pcols)


def testEverything():
  """
  Temporary - for debugging stuff.
  """
  print "Running"
  numpy.random.seed(42)
  tm = MonitoredTemporalMemory(minThreshold=30, activationThreshold=30,
                              maxNewSynapseCount=40, cellsPerColumn=5,
                              learnOnOneCell = False,
                              permanenceOrphanDecrement = 0.005)

  inputVecs = []
  for i in range(4):
    v = getRandomVector()
    inputVecs.append(v)
    print "input i=",v

  print "=================="

  i=0
  while i < 1000:
    if i%100==0:
      print "i=",i
    if i%8 <= 3:
      inputVec = inputVecs[i%4]
    else:
      inputVec = getRandomVector()

    tm.compute(inputVec, learn=True)
    if i >= 1:
      print i,len(tm.predictiveCells)
    i += 1

  # doesn't work?
  #tm.mmGetCellActivityPlot()

  print tm.mmPrettyPrintSequenceCellRepresentations()

  pac = tm.mmGetTracePredictedActiveColumns()
  pic = tm.mmGetTracePredictedInactiveColumns()
  upac = tm.mmGetTraceUnpredictedActiveColumns()

  print len(pac.data),len(pic.data)
  print "i pac pic upac err"
  for i,j in enumerate(pac.data):
    print i,len(j),len(pic.data[i]),len(upac.data[i]),
    print computePredictionAccuracy(len(j), len(pic.data[i]))


def printOptions(options):
  """
  Pretty print the set of options
  """
  print "Experiment parameters:"
  for k,v in options.__dict__.iteritems():
    print "  %s : %s" % (k,str(v))


def runExperiment1(csvWriter, options):
  csvWriter.writerow(["time", "element", "pac", "pic", "upac", "a", "accuracy"])
  numpy.random.seed(42)

  tm = MonitoredTemporalMemory(minThreshold=30,
                              activationThreshold=30,
                              maxNewSynapseCount=40,
                              cellsPerColumn=options.cells,
                              learnOnOneCell = False,
                              permanenceOrphanDecrement = 0.01,
                              columnDimensions=(2048,),
                              cellsPerColumn=32,
                              learningRadius=2048,
                              initialPermanence=0.21,
                              connectedPermanence=0.50,
                              permanenceIncrement=0.10,
                              permanenceDecrement=0.10,
                              seed=42,
                              )

  sequenceString = ""
  i=0
  while i < options.iterations:
    if i%100==0:
      print "i=",i

    if options.simulation == "low":
      vecs,label = getLowOrderSequenceChunk(i)
    elif options.simulation == "high":
      vecs,label = getHighOrderSequenceChunk(i, options.switchover)
    else:
      raise Exception("Unknown simulation: " + options.simulation)

    sequenceString += label

    for xi,vec in enumerate(vecs):
      tm.compute(vec, learn=True)

      # if i >= 1:
      #   print i,label[xi],len(tm.predictiveCells)
      i += 1

  # Print out trace of predictions and accuracy
  pac = tm.mmGetTracePredictedActiveColumns()
  pic = tm.mmGetTracePredictedInactiveColumns()
  upac = tm.mmGetTraceUnpredictedActiveColumns()

  accuracy = 0.0
  print len(pac.data),len(pic.data)
  print "i elmt pac pic upac err"
  for i,j in enumerate(pac.data):
    if i>0:
      a = computePredictionAccuracy(len(j), len(pic.data[i]))
      accuracy = 0.995*accuracy + 0.005*a
      print i,sequenceString[i],len(j),len(pic.data[i]),len(upac.data[i]),
      print a,accuracy
      csvWriter.writerow([i, sequenceString[i],len(j),len(pic.data[i]),
                          len(upac.data[i]), a, accuracy])

if __name__ == '__main__':
  helpString = (
    "\n%prog [options] [uid]"
    "\n%prog --help"
    "\n"
    "\nRuns high and low order sequence simulations with artificial data."
  )

  # All the command line options
  parser = OptionParser(helpString)
  parser.add_option("--outputFile",
                    help="Output file. Results will be written to this file."
                    " (default: %default)",
                    dest="outputFile",
                    default="output.csv")
  parser.add_option("--iterations",
                    help="Number of iterations to run for. [default: %default]",
                    default=1000,
                    type=int)
  parser.add_option("--switchover",
                    help="Number of iterations after which to change "
                         "statistics. [default: %default]",
                    default=1000,
                    type=int)
  parser.add_option("--cells",
                    help="Number of per column. [default: %default]",
                    default=8,
                    type=int)
  parser.add_option("--simulation",
                    help="Which simulation to run: 'low' or 'high'"
                    " (default: %default)",
                    default="low",
                    type=str)

  options, args = parser.parse_args(sys.argv[1:])
  printOptions(options)

  with open(options.outputFile,"wb") as outputFile:
    csvWriter = csv.writer(outputFile)

    #testEverything()
    runExperiment1(csvWriter, options)
