#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

""" Useful scripts for sensor data sequence classification experiments. """

import csv
import os
import math
import random



def generateSensorData(signalType,
                       dataDir,
                       numPhases,
                       numReps,
                       signalMean,
                       signalAmplitude,
                       numCategories,
                       noiseAmplitude):
  """
  Generate the artificial sensor data.
  @param signalType: (str) can be one of the following:
  - 'sine': Sine wave with a different frequency per category.
  - 'binary': periodic binary signal with a different period per category.
  - 'triangle': triangular signal with a different sampling rate per category.
  @param dataDir: (str) directory where to save the CSV files
  @param numPhases: (int) number of phases to train and test. E.g: 4 phases 
    to train and 1 to test: (1) SP, (2) TM, (3) TP, (4) Classifier, (5) Test.
  @param numReps: (int) Number of time each phase repeats.
  @param signalMean: (float) mean of the signal to generate
  @param signalAmplitude: (float) amplitude of the signal to generate
  @param numCategories: (int) number of categories labels
  @param noiseAmplitude: (float) amplitude of the white noise
  @return expSetup: (dict) setup for each experiment
  """

  # some info about the experiment setup
  expSetup = {
    'signalType': signalType,
    'signalMean': signalMean,
    'signalAmplitude': signalAmplitude,
    'numCategories': numCategories,
    'noiseAmplitude': noiseAmplitude,
    'numPhases': numPhases,
    'numReps': numReps,
  }

  if signalType == 'sine':
    signal_generator = sine_wave_generator
  elif signalType == 'binary':
    signal_generator = binary_signal_generator
  elif signalType == 'triangle':
    signal_generator = triangular_signal_generator
  else:
    raise ValueError('Signal type can only be "sine", "triangle" or "step"')

  # make sure the directory exist. if not, create it.
  if not os.path.exists(dataDir):
    os.makedirs(dataDir)

  filePath = "%s/%s_ampl=%s_mean=%s_noise=%s.csv" % (dataDir, signalType,
                                                     signalAmplitude,
                                                     signalMean,
                                                     noiseAmplitude)
  with open(filePath, "wb") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "label"])
    writer.writerow(["float", "float", "int"])
    writer.writerow(["", "", "C"])  # C is for category. 
    # WARNING: if the C flag is forgotten in the dataset, then all records will
    #  be arbitrarily put in the same category (i.e. category 0). So make sure 
    # to have the C flag -- otherwise you'll get 100% classification accuracy 
    # regardless of the input data :-P

    sequenceLength, numPoints = signal_generator(writer,
                                                 numPhases,
                                                 numReps,
                                                 signalMean,
                                                 noiseAmplitude,
                                                 signalAmplitude,
                                                 numCategories)

  expSetup['sequenceLenght'] = sequenceLength
  expSetup['numPoints'] = numPoints
  expSetup['filePath'] = filePath

  return expSetup



def sine_wave_generator(writer,
                        numPhases,
                        numReps,
                        signalMean,
                        noiseAmplitude,
                        signalAmplitude,
                        numCategories):
  """
  Generate the artificial sensor data.
  @param writer: (csv.CSVWriter) csv file writer.
  @param numPhases: (int) number of phases to train and test. E.g: 4 phases 
    to train and 1 to test: (1) SP, (2) TM, (3) TP, (4) Classifier, (5) Test.
  @param numReps: (int) Number of time each phase repeats.
  @param signalMean: (float) mean of the signal to generate
  @param noiseAmplitude: (float) amplitude of the white noise
  @param signalAmplitude: (float) amplitude of the signal to generate
  @param numCategories: (int) number of categories labels
  """

  signalPeriod = 20
  sequenceLength = signalPeriod * 4
  numPoints = numReps * numPhases * sequenceLength * numCategories

  endOfSequence = sequenceLength
  label = numCategories - 1
  for i in range(numPoints):

    noise = noiseAmplitude * random.random()

    if i == endOfSequence:
      endOfSequence += sequenceLength
      if label == 0:
        label = numCategories - 1
      else:
        label -= 1

    signal_modifier = (label ** 2 + 1)
    x = signal_modifier * (i * math.pi) / signalPeriod
    sig = math.sin(x)
    m1 = signal_modifier * signalMean + signalAmplitude * sig + noise

    writer.writerow([x, m1, label])

  return sequenceLength, numPoints



def binary_signal_generator(writer,
                            numPhases,
                            numReps,
                            signalMean,
                            noiseAmplitude,
                            signalAmplitude,
                            numCategories):
  """
  Generate the artificial sensor data.
  @param writer: (csv.CSVWriter) csv file writer.
  @param numPhases: (int) number of phases to train and test. E.g: 4 phases 
    to train and 1 to test: (1) SP, (2) TM, (3) TP, (4) Classifier, (5) Test.
  @param numReps: (int) Number of time each phase repeats.
  @param signalMean: (float) mean of the signal to generate
  @param noiseAmplitude: (float) amplitude of the white noise
  @param signalAmplitude: (float) amplitude of the signal to generate
  @param numCategories: (int) number of categories labels
  """

  # if numCategories = 3, then sequenceLength = 4 * 3 * 2 = 24
  # if numCategories = 2, then sequenceLength = 3 * 2 = 6
  minSequenceLength = 1
  for cat in range(numCategories + 1)[1:]:
    minSequenceLength *= cat
  sequenceLength = minSequenceLength * 3
  numPoints = numReps * numPhases * sequenceLength * numCategories

  endOfSequence = sequenceLength
  label = numCategories - 1
  periodCounter = [0 for _ in range(numCategories)]
  sig = 0
  for i in range(numPoints):
    noise = noiseAmplitude * random.random()

    if i == endOfSequence:
      endOfSequence += sequenceLength
      if label == 0:
        label = numCategories - 1
      else:
        label -= 1

    if periodCounter[label] == label + 1:
      periodCounter = [0 for _ in range(numCategories)]
      if sig == 0:
        sig = 1
      else:
        sig = 0
    else:
      periodCounter[label] += 1

    amplitude_modifier = float(label) ** 2
    m1 = amplitude_modifier * signalMean + signalAmplitude * sig + noise

    writer.writerow([i, m1, label])

  return sequenceLength, numPoints



def triangular_signal_generator(writer,
                                numPhases,
                                numReps,
                                signalMean,
                                noiseAmplitude,
                                signalAmplitude,
                                numCategories):
  """
  Generate the artificial sensor data.
  @param writer: (csv.CSVWriter) csv file writer.
  @param numPhases: (int) number of phases to train and test. E.g: 4 phases 
    to train and 1 to test: (1) SP, (2) TM, (3) TP, (4) Classifier, (5) Test.
  @param numReps: (int) Number of time each phase repeats.
  @param signalMean: (float) mean of the signal to generate
  @param noiseAmplitude: (float) amplitude of the white noise
  @param signalAmplitude: (float) amplitude of the signal to generate
  @param numCategories: (int) number of categories labels
  """

  # if numCategories = 3, then sequenceLength = 4 * 3 * 2 = 24
  # if numCategories = 2, then sequenceLength = 3 * 2 = 6
  minSequenceLength = math.factorial(numCategories + 1)
  sequenceLength = minSequenceLength * 3
  numPoints = numReps * numPhases * sequenceLength * numCategories

  endOfSequence = sequenceLength
  label = numCategories - 1
  for i in range(numPoints):

    noise = noiseAmplitude * random.random()

    if i == endOfSequence:
      endOfSequence += sequenceLength
      if label == 0:
        label = numCategories - 1
      else:
        label -= 1

    x = i
    mod = int(label) + 2
    # if label = 0, then mod = 2, so sig = 0 or 1
    # if label = 1, then mod = 3, so sig = 0 or 0.5 or 1
    # if label = 2, then mod = 4, so sig = 0 or 0.33 or 0.66 or 1
    sig = i % mod / (mod - 1.0)

    amplitude_modifier = float(label) ** 2
    m1 = amplitude_modifier * signalMean + signalAmplitude * sig + noise

    writer.writerow([x, m1, label])

  return sequenceLength, numPoints
