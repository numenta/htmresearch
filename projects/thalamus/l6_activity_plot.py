#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

"""
This file plots activity of L6 cells over time
"""

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from htmresearch.frameworks.thalamus.thalamus import Thalamus
from htmresearch.frameworks.thalamus.thalamus_utils import *

plotlyUser = os.environ['PLOTLY_USERNAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']
py.sign_in(plotlyUser, plotlyAPIKey)


def plotL6SDRs(encoder, t):
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)

  shapes = []
  for x in range(0,32):
    activeCells = encodeLocation(encoder, x, x, output)
    for cell in activeCells:
      shapes.append(
        {
          'type': 'rect',
          'x0': x,
          'x1': x + 0.75,
          'y0': cell,
          'y1': cell + 2,
          'line': {
            # 'color': 'rgba(128, 0, 128, 1)',
            'width': 2,
          },
          # 'fillcolor': 'rgba(128, 0, 128, 0.7)',
        },
      )

  data = [go.Scatter(x=[], y=[])]
  layout = {
    'width': 600,
    'height': 600,
    'font': {'size': 20},
    'xaxis': {
      'title': "Location",
      'range': [0, t.trnWidth],
      'showgrid': False,
      'showticklabels': True,
    },
    'yaxis': {
      'title': "Neuron #",
      'range': [-100, 1024],
      'showgrid': False,
    },
    'shapes': shapes,
    'annotations': [ {
      'xanchor': 'center',
      'yanchor': 'bottom',
      'text': 'Target object',
      'x': 1,
      'y': 4100,
      'ax': 10,
      'ay': -25,
      'arrowcolor': 'rgba(255, 0, 0, 1)',
      },
      {
        'font': {'size': 24},
        'xanchor': 'center',
        'yanchor': 'bottom',
        'text': '<b>L6a activity for locations along diagonal</b>',
        'xref': 'paper',
        'yref': 'paper',
        'x': 0.5,
        'y': 1.1,
        'showarrow': False,
      }
    ]
  }
  fig = {
    'data': data,
    'layout': layout,
  }
  plotPath = plotly.offline.plot(fig, filename='temp.html',
                                  auto_open=True)
  print("url=", plotPath)

  # Can't save image files in offline mode
  plotly.plotly.image.save_as(
    fig, filename=os.path.join("images", "L6_activity.pdf"), scale=1)


if __name__ == "__main__":
  t = Thalamus()
  encoder = createLocationEncoder(t)
  plotL6SDRs(encoder, t)
