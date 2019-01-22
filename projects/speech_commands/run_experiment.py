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

import matplotlib
matplotlib.use('Agg')
import tarfile
from os import path
import os
import shutil

from htmresearch.frameworks.pytorch.sparse_speech_experiment import \
  SparseSpeechExperiment


# Run from root dir:
# domino run projects/speech_commands//run_experiment.py -c projects/speech_commands/experiments.cfg
#   OR
# python projects/speech_commands/run_experiment.py -c projects/speech_commands/experiments.cfg -d
if __name__ == '__main__':
  suite = SparseSpeechExperiment()

  suite.parse_opt()
  projectDir = path.dirname(suite.options.config)
  dataDir = path.join(path.abspath(projectDir), "data")

  if "DOMINO_WORKING_DIR" in os.environ:
    print("In domino")
    if path.isdir(path.join(dataDir, "speech_commands")):
      print("Removing speech_commands")
      shutil.rmtree(path.join(dataDir, "speech_commands"))

    if not path.isdir(path.join(dataDir, "speech_commands")):
      print("Untarring dataset...")
      tar = tarfile.open(path.join(dataDir, "speech_commands.tar.gz"))
      tar.extractall(path=dataDir)
      tar.close()
    else:
      print("Apparently this still exists:", path.join(dataDir, "speech_commands"))

    print("listing files:")
    files = os.listdir(path.join(dataDir, "speech_commands"))
    print(files)
    files = os.listdir(path.join(dataDir, "speech_commands", "train", "one"))
    print("train/one", len(files), files[0:10])

  suite.start()
