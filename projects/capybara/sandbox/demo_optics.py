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
Install pyclustering: 
    git clone https://github.com/annoviko/pyclustering
    cd pyclustering
    pip install . --user
"""
import numpy as np
import random as rd
import colorlover as cl

from pyclustering.cluster.optics import optics, ordering_analyser
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

  sample = np.array(sample)
  np.random.shuffle(sample)
  sample = sample.tolist()
  return sample



# Read sample for clustering from some file (2D matrix)
sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
# sample = gaussian_clusters(3)

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

# Find valleys boundaries
cluster_ordering = np.array(ordering.cluster_ordering)
peaks = np.where(cluster_ordering > epsilon_cutoff)[0]
val_boundaries = [int(i) for i in peaks]
val_boundaries.insert(0, 0)
val_boundaries.append(len(ordering.cluster_ordering))
print 'valleys boundaries: %s' % val_boundaries
num_valleys = len(val_boundaries) - 1

# Plot input data and clustering structure

# ordering_visualizer.show_ordering_diagram(ordering)

# Save and look at the color palette :-)
color_palette = cl.scales['3']  # 3 colors
palette_file = 'palette.html'
with open(palette_file, 'w+') as f:
  f.write(cl.to_html(color_palette))
  print 'color palette saved to: %s' % palette_file
colors = color_palette['qual']['Set1']

# Data traces to plot
traces1 = [go.Scatter(
  name='Cluster %s data' % i,
  x=[sample[sample_id][0] for sample_id in
     clusters[0][val_boundaries[i] + 1:val_boundaries[i + 1]]],
  y=[sample[sample_id][1] for sample_id in
     clusters[0][val_boundaries[i] + 1:val_boundaries[i + 1]]],
  mode='markers',
  marker=dict(color=colors[i])
) for i in range(num_valleys)]

traces2 = [go.Scatter(
  name='Cluster %s reachability distances' % i,
  x=range(val_boundaries[i], val_boundaries[i + 1] + 1),
  y=ordering.cluster_ordering[val_boundaries[i]:val_boundaries[i + 1] + 1],
  mode='lines',
  line=dict(color=colors[i])
) for i in range(num_valleys)]

traces2.append(go.Scatter(
  name='Reachability threshold',
  y=[epsilon_cutoff for _ in ordering.cluster_ordering],
  mode='lines',
  line=dict(color='grey', dash='dash')
))

layout1 = go.Layout(title='Input data',
                    xaxis=go.XAxis(title='x', range=[0, 6]),
                    yaxis=go.YAxis(title='y', range=[0, 6]))
layout2 = go.Layout(title='Clustering structure',
                    xaxis=go.XAxis(title='Index (cluster order of the '
                                         'objects)'),
                    yaxis=go.YAxis(title='Reachability distance',
                                   range=[0, 1]),
                    annotations=[
                      dict(
                        x=val_boundaries[i],
                        y=ordering.cluster_ordering[val_boundaries[i]],
                        xref='x',
                        yref='y',
                        text='Reachability distance over threshold',
                        showarrow=True,
                        arrowhead=7,
                        ax=0,
                        ay=-40
                      ) for i in range(1, num_valleys)]
                    )

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

description = """
<b><a href="https://en.wikipedia.org/wiki/OPTICS_algorithm">OPTICS
</a> 2D Example</b>
"""

fig3['layout'].update(width=1200, height=600, title=description)

url3 = plot(fig3, filename='clustering-structure.html', auto_open=False)
print url3
