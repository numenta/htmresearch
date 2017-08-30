#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
import argparse
import os
from subprocess import call

# Parse input options.
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', '-o',
                  dest='output_dir',
                  default=os.path.join(os.getcwd()),
                  type=str)

options = parser.parse_args()
output_dir = options.output_dir

print("Downloading...")
if not os.path.exists("%s/UCI HAR Dataset.zip" % output_dir):
    call('wget "https://archive.ics.uci.edu/ml/machine-learning-databases/'
         '00240/UCI%20HAR%20Dataset.zip" -P ' + output_dir, shell=True)
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")


print("Extracting...")
extract_directory = os.path.abspath(os.path.join(output_dir, 
                                                 "UCI HAR Dataset"))
if not os.path.exists(extract_directory):
    call('unzip -nq "%s/UCI HAR Dataset.zip" -d %s' % (output_dir,
                                                       output_dir), 
                                                       shell=True)
    print("Extracting successfully done to {}.".format(extract_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")
