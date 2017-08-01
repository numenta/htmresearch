# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

# This uses plotly to create a nice looking graph of average false positive
# error rates as a function of N, the dimensionality of the vectors.  I'm sorry
# this code is so ugly.

import plotly.plotly as py
from plotly.graph_objs import *
import os

plotlyUser = os.environ['PLOTLY_USERNAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']


py.sign_in(plotlyUser, plotlyAPIKey)


# Observed vs theoretical error values

# a=64 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimentalErrorsA64 = [2736527./20000000,
                         621./10000000,
                         2./10000000,
                         0./10000000]

# a=128 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimentalErrorsA128 = [959346./2000000,
                          692581./2000000,
                          36401./1000000,
                          745./1000000,
                          236./11000000,
                          11./11000000,
                          3./11000000,
                          1./11000000,
                          0./11000000,
                          0./1000000,
                          0./1000000,
                          0./1000000]

# a=256 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimentalErrorsA256 = [1001193./2000000,
                          979520./2000000,
                          464716./1000000,
                          406276./1000000,
                          276204./1000000,
                          96118./1000000,
                          17508./1000000,
                          2506./1000000,
                          387./1000000,
                          707./11000000,
                          165./11000000,
                          31./11000000,
                          7./11000000,
                          4./11000000,
                          0./1000000]

listofNValues = [300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300,
2500, 2700, 2900, 3100, 3300, 3500]

trace1 = Scatter(
    y=experimentalErrorsA64,
    x=listofNValues[0:3],
    mode="lines + markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="a=64 (observed)"
)

trace2 = Scatter(
    y=experimentalErrorsA128,
    x=listofNValues[0:9],
    mode="lines + markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="a=128 (observed)"
)


trace3 = Scatter(
    y=experimentalErrorsA256,
    x=listofNValues,
    mode="lines+markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="a=256 (observed)"
)


data = Data([trace1, trace2, trace3])

layout = Layout(
    title='',
    showlegend=False,
    autosize=False,
    width=855,
    height=700,
    xaxis=XAxis(
        title='Cell population size (n)',
        titlefont=Font(
            family='',
            size=26,
            color=''
        ),
        tickfont=Font(
            family='',
            size=16,
            color=''
        ),
        exponentformat="none",
        dtick=400,
        showline=True,
        range=[0,3500],
    ),
    yaxis=YAxis(
        title='Frequency of false positives',
        type='log',
        exponentformat='power',
        autorange=True,
        titlefont=Font(
            family='',
            size=26,
            color=''
        ),
        tickfont=Font(
            family='',
            size=12,
            color=''
        ),
        showline=True,
    ),
    annotations=Annotations([
      Annotation(
            x=434,
            y=0.2085,
            xref='x',
            yref='paper',
            text='$a = 64$',
            showarrow=False,
            font=Font(
                family='',
                size=16,
                color=''
            ),
            align='center',
            textangle=0,
            bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
      Annotation(
            x=1776,
            y=0.2071,
            xref='x',
            yref='paper',
            text='$a = 128$',
            showarrow=False,
            font=Font(
                family='',
                size=16,
                color=''
            ),
            align='center',
            textangle=0,
            bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
      Annotation(
            x=3272,
            y=0.2014,
            xref='x',
            yref='paper',
            text='$a = 256$',
            showarrow=False,
            font=Font(
                family='',
                size=16,
                color=''
            ),
            align='center',
            textangle=0,
            bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
    ]),)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
print "url=",plot_url
figure = py.get_figure(plot_url)
py.image.save_as(figure, 'images/effect_of_n.png', scale=4)
