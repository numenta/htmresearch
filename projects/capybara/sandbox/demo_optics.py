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
Install: 
    git clone https://github.com/annoviko/pyclustering
    cd pyclustering
    pip install . --user
"""
import numpy as np
import random as rd

from pyclustering.cluster.optics import optics, ordering_analyser, \
  ordering_visualizer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample

from plotly.offline import plot
from plotly.tools import make_subplots
import plotly.graph_objs as go



def gaussian_clusters(num_clusters):
  sample = []
  for i in range(1, num_clusters + 1):
    for _ in range(40):
      sample.append([i * 1.5 + rd.random(), i * 1.5 + rd.random()])
  return sample



# Read sample for clustering from some file
# 2D matrix
# sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)[0:95]
# sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)[100:190]
# sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)[200:402]
# sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
sample = gaussian_clusters(3)

# Run cluster analysis where connectivity radius is bigger than real
radius = 2.0
neighbors = 2

# Create OPTICS algorithm for cluster analysis
optics_instance = optics(sample, radius, neighbors)

# Run cluster analysis
optics_instance.process()

# Obtain results of clustering
clusters = optics_instance.get_clusters()
noise = optics_instance.get_noise()

# Obtain reachability-distances
ordering = ordering_analyser(optics_instance.get_ordering())

# Cluster assignments
epsilon_cutoff = 0.5

cluster_ordering = np.array(ordering.cluster_ordering)
valleys = np.where(cluster_ordering < epsilon_cutoff)
val = np.where(np.diff(valleys[0]) > 1)[0] + 1
val_boundaries = [int(i) for i in val]
val_boundaries.insert(0, 0)
val_boundaries.append(len(ordering.cluster_ordering))
print 'valleys boundaries: %s' % val_boundaries
num_valleys = len(val_boundaries) - 1
# Visualization of cluster ordering in line with reachability distance.
# ordering_visualizer.show_ordering_diagram(ordering)


# Plot input data and clustering structure
colors = ['r', 'g', 'b']

traces1 = [go.Scatter(
  name='Cluster %s' % i,
  # buffer of points (before and after) around the peak
  x=[s[0] for s in sample[val_boundaries[i] + 2:val_boundaries[i + 1] - 1]],
  y=[s[1] for s in sample[val_boundaries[i] + 2:val_boundaries[i + 1] - 1]],
  mode='markers',
  marker=dict(color=colors[i])
) for i in range(num_valleys)]

traces2 = [go.Scatter(
  name='Cluster %s' % i,
  x=range(val_boundaries[i] + 2, val_boundaries[i + 1] - 1),
  y=ordering.cluster_ordering[val_boundaries[i] + 2:val_boundaries[i + 1] - 1],
  mode='lines',
  line=dict(color=colors[i])
) for i in range(num_valleys)]

traces2.extend([go.Scatter(
  name='Not included in clusters',
  x=range(val_boundaries[i] - 2, val_boundaries[i] + 3),
  y=ordering.cluster_ordering[val_boundaries[i] - 2:val_boundaries[i] + 3],
  mode='lines',
  line=dict(color='black')
) for i in range(1, num_valleys)])

layout1 = go.Layout(title='Input data',
                    xaxis=go.XAxis(title='x'),
                    yaxis=go.YAxis(title='y'))

layout2 = go.Layout(title='Clustering structure',
                    xaxis=go.XAxis(title='Index (cluster order of the '
                                         'objects)'),
                    yaxis=go.YAxis(title='Reachability distance'))

fig1 = {'data': traces1, 'layout': layout1}
url1 = plot(fig1, auto_open=False, filename='data.html')
print url1

fig2 = {'data': traces2, 'layout': layout2}
url2 = plot(fig2, auto_open=False, filename='reachability.html')
print url2

# With subplots:
fig3 = make_subplots(rows=1, cols=2, subplot_titles=['Input Data',
                                                     'Clustering Structure'])
for trace1 in traces1:
  fig3.append_trace(trace1, row=1, col=1)
for trace2 in traces2:
  fig3.append_trace(trace2, row=1, col=2)

# Subplot layout example:
fig3['layout'].update(xaxis1=go.XAxis(title='x'),
                      yaxis1=go.YAxis(title='y'))
# Or:
fig3['layout']['xaxis1'].update(title='X values', range=[0, 6])
fig3['layout']['yaxis1'].update(title='Y values', range=[0, 6])
fig3['layout']['xaxis2'].update(title='Index (cluster order of the objects)')
fig3['layout']['yaxis2'].update(title='Reachability distance', range=[0, 1])

fig3['layout'].update(width=1200, height=600, title='OPTICS 2D Example')

url3 = plot(fig3, filename='clustering-structure.html', auto_open=False)
print url3
