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

"""Plot simulation results."""

import os

import plotly.plotly as py
from plotly.graph_objs import Scatter, Figure, Layout


def signin():
  py.sign_in(os.environ["PLOTLY_USERNAME"], os.environ["PLOTLY_API_KEY"])


def getPlot(y, symbol, dash, ytitle):
  return Scatter(**{
      "x": x,
      "y": y,
      "name": ytitle,
      "marker": {
          "symbol": symbol,
          "size": 10,
      },
      "line": {
          "dash": dash,
          "color": "black",
      },
  })


layout = {
    "xaxis": {
        "title": "Noise (# of swapped landmarks)",
        "range": [-0.5, 16.5],
        "titlefont": {
            "size": 16,
        },
    },
    "yaxis": {
        "title": "Recall Accuracy",
        "range": [-0.05, 1.05],
        "titlefont": {
            "size": 16,
        },
    },
    "margin": {
        "t": 30,
        "r": 30,
    },
    "legend": {
        "x": 0.8,
        "borderwidth": 2,
        "font": {
            "size": 16,
        },
    },
}


x = [i*2 for i in xrange(9)]

# Generated with following parameters:
# TM with 4 cells per column, 2048 columns, 20 active
#   self.l4TM = TemporalMemory(
#       columnCount=l4N,
#       basalInputSize=numModules*self._cellsPerModule,
#       cellsPerColumn=4,
#       #activationThreshold=int(numModules / 2) + 1,
#       #reducedBasalThreshold=int(numModules / 2) + 1,
#       activationThreshold=1,
#       reducedBasalThreshold=1,
#       initialPermanence=1.0,
#       connectedPermanence=0.5,
#       minThreshold=1,
#       sampleSize=numModules,
#       permanenceIncrement=1.0,
#       permanenceDecrement=0.0,
#   )
#   objectDims = (4, 4)
#
#   numTrainingPasses = 5
#   numTestingPasses = 3
#   maxActivePerModule = 25
#   skipFirst = 5
#   numModules = 2
#   moduleDims = (50, 50)
#
#   l4N = 2048
#   l4W = 40
#   l6ActivationThreshold = 8
#   1000 objects

# 5 possible features
sim5 = [1, 1, 0.992, 0.875, 0.495, 0.162, 0.029, 0.005, 0.001]
# From simulations
ideal5 = [1.0, 1.0, 0.999425, 0.958525, 0.610675, 0.21245, 0.04685, 0.008075, 0.001]
# Simulations showed this is no better than chance
bof5 = [0.001] * 9

# 50 possible features (not used in the CNS submission)
sim50 = [1.0, 1.0, 1.0, 0.998, 0.939, 0.456, 0.086, 0.0, 0.001]
ideal50 = [1, 1, 1, 1, 1, 0.999775, 0.907775, 0.074875, 0.001]
bof50 = [1, 0.9996, 0.980875, 0.782, 0.347525, 0.0958, 0.0231, 0.0048, 0.001]

fig = Figure(
    data = [
      getPlot(bof5, "triangle-up-open", "dash", "5 (BOF)"),
      getPlot(sim5, "circle", "solid", "5"),
      getPlot(ideal5, "square-open", "dash", "5 (Ideal)"),
      #getPlot(sim50, "square-open", "dash", "50"),
      #getPlot(ideal50, "circle", "dash", "50 (Ideal)"),
      #getPlot(bof50, "triangle-up", "dash", "50 (BOF)"),
    ],
    layout=layout,
)

signin()
py.image.save_as(fig, "accuracy_plot.pdf", width=700, height=400)
