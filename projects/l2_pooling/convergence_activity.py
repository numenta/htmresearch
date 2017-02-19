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

"""
This file plots activity of single vs multiple columns as they converge.
"""

import random
import os
import pprint
import plotly
import plotly.graph_objs as go

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)

plotlyUser = os.environ['PLOTLY_USERNAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']
plotly.plotly.sign_in(plotlyUser, plotlyAPIKey)


def plotActivity(l2ActiveCellsMultiColumn):
  maxTouches = 15
  numTouches = min(maxTouches, len(l2ActiveCellsMultiColumn))
  numColumns = len(l2ActiveCellsMultiColumn[0])
  fig = plotly.tools.make_subplots(
          rows=1, cols=numColumns, shared_yaxes=True,
    subplot_titles=('Column 1', 'Column 2', 'Column 3')[0:numColumns]
  )
  # pprint.pprint(fig)

  data = go.Scatter(x=[], y=[])

  shapes = []
  for t,sdrs in enumerate(l2ActiveCellsMultiColumn):
    if t <= numTouches:
      for c, activeCells in enumerate(sdrs):
        # print t, c, len(activeCells)
        for cell in activeCells:
          shapes.append(
            {
              'type': 'rect',
              'xref': 'x'+str((c+1)),
              'yref': 'y1',
              'x0': t,
              'x1': t + 0.6,
              'y0': cell,
              'y1': cell + 1,
              'line': {
                # 'color': 'rgba(128, 0, 128, 1)',
                'width': 2,
              },
              # 'fillcolor': 'rgba(128, 0, 128, 0.7)',
            },
          )

  # Add red rectangle
  if numColumns==1:
    shapes.append(
      {
        'type': 'rect',
        'x0': 11,
        'x1': 11.6,
        'y0': -95,
        'y1': 4100,
        'line': {
          'color': 'rgba(255, 0, 0, 0.5)',
          'width': 3,
        },
      },
    )
  else:
    shapes.append(
      {
        'type': 'rect',
        'x0': 3,
        'x1': 3.6,
        'y0': -95,
        'y1': 4100,
        'line': {
          'color': 'rgba(255, 0, 0, 0.5)',
          'width': 3,
        },
      },
    )


  # Legend for x-axis and appropriate title
  fig['layout']['annotations'].append({
    'font': {'size': 16},
    'xanchor': 'center',
    'yanchor': 'bottom',
    'text': 'Number of touches',
    'xref': 'paper',
    'yref': 'paper',
    'x': 0.5,
    'y': -0.1,
    'showarrow': False,
  })
  fig['layout']['annotations'].append({
    'font': {'size': 16},
    'xanchor': 'center',
    'yanchor': 'bottom',
    'text': ['','One cortical column','','Three cortical columns'][numColumns],
    'xref': 'paper',
    'yref': 'paper',
    'x': 0.5,
    'y': 1.1,
    'showarrow': False,
  })
  layout = {
    'height': 600,
    'yaxis': {
      'title': "Neuron #",
      'range': [-100, 4201],
      'showgrid': False,
    },
    'shapes': shapes,
  }


  if numColumns == 1: layout.update(width=320)
  else: layout.update(width=700)

  for c in range(numColumns):
    fig.append_trace(data, 1, c+1)
    fig['layout']['xaxis'+str(c+1)].update({
      'title': "",
      'range': [0, numTouches],
      'showgrid': False,
      'showticklabels': True,
    }),

  fig['layout'].update(layout)

  # Save plots as HTM and/or PDF
  basename='plots/activity_c'+str(numColumns)
  plotly.offline.plot(fig, filename=basename+'.html', auto_open=True)

  # Can't save image files in offline mode
  plotly.plotly.image.save_as(fig, filename=basename+'.pdf', scale=4)



def plotL2ObjectRepresentations(exp1):
  shapes = []
  numObjects = len(exp1.objectL2Representations)
  for obj in range(numObjects):
    activeCells = exp1.objectL2Representations[obj][0]
    for cell in activeCells:
      shapes.append(
        {
          'type': 'rect',
          'x0': obj,
          'x1': obj + 0.75,
          'y0': cell,
          'y1': cell + 2,
          'line': {
            # 'color': 'rgba(128, 0, 128, 1)',
            'width': 2,
          },
          # 'fillcolor': 'rgba(128, 0, 128, 0.7)',
        },
      )

  # Add red rectangle
  shapes.append(
    {
      'type': 'rect',
      'x0': 0,
      'x1': 0.9,
      'y0': -95,
      'y1': 4100,
      'line': {
        'color': 'rgba(255, 0, 0, 0.5)',
        'width': 3,
      },
    },
  )


  data = [go.Scatter(x=[], y=[])]
  layout = {
    'width': 320,
    'height': 600,
    'xaxis': {
      'title': "Object #",
      'range': [0, 10],
      'showgrid': False,
      'showticklabels': True,
    },
    'yaxis': {
      'title': "Neuron #",
      'range': [-100, 4201],
      'showgrid': False,
    },
    'shapes': shapes,
    'annotations': [ {
      'xanchor': 'middle',
      'yanchor': 'bottom',
      'text': 'Target object',
      'x': 1,
      'y': 4100,
      'ax': 10,
      'ay': -25,
      'arrowcolor': 'rgba(255, 0, 0, 1)',
      },
      {
        'font': {'size': 16},
        'xanchor': 'center',
        'yanchor': 'bottom',
        'text': 'Object representations',
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
  plotPath = plotly.offline.plot(fig, filename='plots/shapes-rectangle.html',
                                  auto_open=True)
  print "url=", plotPath

  # Can't save image files in offline mode
  plotly.plotly.image.save_as(
    fig, filename='plots/target_object_representations.pdf', scale=4)


if __name__ == "__main__":
  numColumns = 3
  numFeatures = 10
  numPoints = 10
  numLocations = 10
  numObjects = 10
  numRptsPerSensation = 2

  objectMachine = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=3,
    seed=40,
  )
  objectMachine.createRandomObjects(numObjects, numPoints=numPoints,
                              numLocations=numLocations,
                              numFeatures=numFeatures)

  objects = objectMachine.provideObjectsToLearn()

  # single-out the inputs to the column #1
  objectsSingleColumn = {}
  for i in range(numObjects):
    featureLocations = []
    for j in range(numLocations):
      featureLocations.append({0: objects[i][j][0]})
    objectsSingleColumn[i] = featureLocations

  # we will run two experiments side by side, with either single column
  # or 3 columns
  exp3 = L4L2Experiment(
    'three_column',
    numCorticalColumns=3,
    seed=1
  )

  exp1 = L4L2Experiment(
    'single_column',
    numCorticalColumns=1,
    seed=1
  )

  print "train single column "
  exp1.learnObjects(objectsSingleColumn)
  print "train multi-column "
  exp3.learnObjects(objects)

  # test on the first object
  objectId = 0
  obj = objectMachine[objectId]

  # Create sequence of sensations for this object for all columns
  objectSensations = {}
  random.seed(10)
  for c in range(numColumns):
    objectCopy = [pair for pair in obj]
    random.shuffle(objectCopy)
    # stay multiple steps on each sensation
    sensations = []
    for pair in objectCopy:
      for _ in xrange(numRptsPerSensation):
        sensations.append(pair)
    objectSensations[c] = sensations

  sensationStepsSingleColumn = []
  sensationStepsMultiColumn = []
  for step in xrange(len(objectSensations[0])):
    pairs = [
      objectSensations[col][step] for col in xrange(numColumns)
      ]
    sdrs = objectMachine._getSDRPairs(pairs)
    sensationStepsMultiColumn.append(sdrs)
    sensationStepsSingleColumn.append({0: sdrs[0]})

  print "inference: multi-columns "
  exp3.sendReset()
  l2ActiveCellsMultiColumn = []
  L2ActiveCellNVsTimeMultiColumn = []
  for sensation in sensationStepsMultiColumn:
    exp3.infer([sensation], objectName=objectId, reset=False)
    l2ActiveCellsMultiColumn.append(exp3.getL2Representations())
    activeCellNum = 0
    for c in range(numColumns):
      activeCellNum += len(exp3.getL2Representations()[c])
    L2ActiveCellNVsTimeMultiColumn.append(activeCellNum/numColumns)

  print "inference: single column "
  exp1.sendReset()
  l2ActiveCellsSingleColumn = []
  L2ActiveCellNVsTimeSingleColumn = []
  for sensation in sensationStepsSingleColumn:
    exp1.infer([sensation], objectName=objectId, reset=False)
    l2ActiveCellsSingleColumn.append(exp1.getL2Representations())
    L2ActiveCellNVsTimeSingleColumn.append(len(exp1.getL2Representations()[0]))

  plotActivity(l2ActiveCellsMultiColumn)
  plotActivity(l2ActiveCellsSingleColumn)
  plotL2ObjectRepresentations(exp1)
