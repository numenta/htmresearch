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


from matplotlib import pyplot as plt
plt.ion()

from suite import Suite
from errorMetrics import *
import pandas as pd

from pylab import rcParams
from plot import loadAndPlot
import plotly.plotly as py

rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.facecolor': 'white'})
rcParams.update({'ytick.labelsize': 8})
rcParams.update({'figure.figsize': (12, 6)})

window = 480
figPath = './result/'

plt.close('all')

fig = plt.figure(1)
loadAndPlot('results/nyc_taxi_experiment_one_shot/', window)
loadAndPlot('results/nyc_taxi_experiment_continuous/learning_window5001.0/', window)
plt.legend()
plt.savefig(figPath + 'continuousVsbatch.pdf')

fig = plt.figure(2)
loadAndPlot('results/nyc_taxi_experiment_continuous/learning_window1001.0/', window)
loadAndPlot('results/nyc_taxi_experiment_continuous/learning_window3001.0/', window)
loadAndPlot('results/nyc_taxi_experiment_continuous/learning_window5001.0/', window)
plt.legend()
plt.savefig(figPath + 'continuous.pdf')


fig = plt.figure(3)
loadAndPlot('results/nyc_taxi_experiment_perturb/learning_window1001.0/', window)
loadAndPlot('results/nyc_taxi_experiment_perturb/learning_window3001.0/', window)
loadAndPlot('results/nyc_taxi_experiment_perturb/learning_window5001.0/', window)
plt.legend()
plt.savefig(figPath + 'continuous_perturb.pdf')
# plot_url = py.plot_mpl(fig)
