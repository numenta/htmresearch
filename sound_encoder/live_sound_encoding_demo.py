#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave

from sound_encoder import SoundEncoder


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10



def getAudioStream():
  p = pyaudio.PyAudio()
  return p.open(format=FORMAT,
				channels=CHANNELS,
				rate=RATE,
				input=True,
				frames_per_buffer=CHUNK)


def transformData(data, window):
  return np.array(wave.struct.unpack("%dh"%(len(data)/CHANNELS),\
                                            data))*window


def visualizeSDRs(sdrs):
  sdrsToVisualize = []

  for sdr in sdrs:
    sdrsToVisualize.append([255 if x else 0 for x in sdr])

  imageArray = np.rot90(np.array(sdrsToVisualize))
  plt.imshow(imageArray, cmap='Greys',  interpolation='nearest')
  plt.show()

def recordAndEncode(stream, soundEncoder):

  window = np.blackman(CHANNELS*CHUNK)
  sdrs = []

  print "---recording---"
  for _ in range(0, (RATE/CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    transformedData = transformData(data, window)
    sdr = soundEncoder.encode(transformedData)
    sdrs.append(sdr)

  stream.stop_stream()
  stream.close()
  print "---done---"
  return sdrs


if __name__ == "__main__":
  n = 300
  w = 31
  minval = 20
  maxval = 10000

  soundEncoder = SoundEncoder(n, w, RATE, CHUNK, minval, maxval)
  stream = getAudioStream()
  sdrs = recordAndEncode(stream, soundEncoder)
  visualizeSDRs(sdrs)

