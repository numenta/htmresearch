# ----------------------------------------------------------------------
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

from __future__ import print_function

import os
from torchvision import transforms

from htmresearch.frameworks.pytorch.speech_commands_dataset import (
  SpeechCommandsDataset, BackgroundNoiseDataset
)
from htmresearch.frameworks.pytorch.audio_transforms import *


def save_examples(noise = 0.0, maxVal = 1.0, dataDir="data"):
  """
  Generate sample noise files for listening and debugging.
  :param noise: noise value for the addNoise transform
  :param maxVal: maxVal for the addNoise transform
  :param dataDir: root dir containing speech_commands directory
  """
  testDataDir = os.path.join(dataDir, "speech_commands", "test")
  outDir = os.path.join(dataDir, "noise_examples")
  if not os.path.exists(outDir):
    os.mkdir(outDir)

  # Create noise dataset with noise transform
  noiseTransform = transforms.Compose([
    LoadAudio(),
    FixAudioLength(),
    AddNoise(noise, maxVal=maxVal),
  ])

  noiseDataset = SpeechCommandsDataset(
    testDataDir, noiseTransform, silence_percentage=0,
  )

  for i in range(0, 2552, 100):
    d = noiseDataset[i]
    fname = os.path.join(outDir, noiseDataset.classes[d["target"]] + "_" +
                         str(i) + "_"
                         + str(int(noise*100)) + "_"
                         + str(int(maxVal*100))
                         + ".wav"
                         )
    print(d["path"], fname)
    librosa.output.write_wav(fname, d['samples'], d["sample_rate"])

if __name__ == '__main__':
  for maxVal in [1.0, 0.5, 0.25]:
    for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]:
      save_examples(noise, maxVal)
