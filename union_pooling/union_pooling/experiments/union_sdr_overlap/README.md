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


# Use the following commands to run the union_sdr_overlap experiment

# create directory to save union SDR traces
mkdir results

# run union SDR on un-trained TM
python union_sdr_overlap_experiment.py params/2048_baseline/0_trainingPasses.yaml results/

# run union SDR on trained TM
python union_sdr_overlap_experiment.py params/2048_baseline/5_trainingPasses.yaml results/

# plot result, the result figure will be located in the 'figures/'' directory
python plot_experiment.py --input results/ --csvOutput csvoutput/ --imgOutput figures/
