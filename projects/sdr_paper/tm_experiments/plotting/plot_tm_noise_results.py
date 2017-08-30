# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

# This uses plotly to create a nice looking graph of average false positive
# error rates as a function of N, the dimensionality of the vectors.  I'm sorry
# this code is so ugly.

import plotly.plotly as py
from plotly.graph_objs import *
import os
import numpy

plotlyUser = os.environ['PLOTLY_USERNAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']


py.sign_in(plotlyUser, plotlyAPIKey)

# Observed vs theoretical error values
predictedAccuracyThreshold8 = [1,
1,
1,
1,
1,
1,
1,
0.999931522178463,
0.998471775446191,
0.987579139811088,
0.943583112258452,
0.829678142620059,
0.626236358716893,
0.370526811572418,
0.150422391245528,
0.0324165808036776,
0.00163800163800165,
0,
0,
0]
predictedAccuracyThreshold12 = [1,
1,
1,
1,
1,
0.995819132915907,
0.958820611490244,
0.839867119622403,
0.626236358716893,
0.375593153728280,
0.171533612376094,
0.0554847685137378,
0.0112387136858773,
0.00109982062151137,
2.25475753841131e-5,
0,
0,
0,
0,
0]
predictedAccuracyThreshold16 = [1,
1,
1,
0.909117909117909,
0.652617652617653,
0.358213277568116,
0.150422391245528,
0.0479227837403589,
0.0112387136858773,
0.00182051352433898,
0.000179983683393226,
8.32190704991298e-6,
7.70857278009629e-8,
0,
0,
0,
0,
0,
0,
0]

# a=40 cells active, s=20 synapses on segment, 2048 columns, 1 cell/column
experimentalAccuracyThreshold8 = [1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
0.998684210526,
0.998684210526,
0.988157894737,
0.946052631579,
0.834210526316,
0.614473684211,
0.431578947368,
0.186842105263,
0.0578947368421,
0.00789473684211,
0.0,
0.0,
0.0]

experimentalAccuracyThreshold12 = [1.0,
1.0,
1.0,
1.0,
1.0,
0.994736842105,
0.963157894737,
0.847368421053,
0.660526315789,
0.392105263158,
0.172368421053,
0.0671052631579,
0.0157894736842,
0.00131578947368,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0]

experimentalAccuracyThreshold16 = [1.0,
1.0,
1.0,
0.921052631579,
0.684210526316,
0.380263157895,
0.167105263158,
0.0434210526316,
0.00921052631579,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0]

listofNoiseValues = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

trace1 = Scatter(
    y=experimentalAccuracyThreshold8,
    x=listofNoiseValues,
    mode="markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="theta=8 (observed)"
)

trace2 = Scatter(
    y=experimentalAccuracyThreshold12,
    x=listofNoiseValues,
    mode="markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="theta=12 (observed)"
)

trace3 = Scatter(
    y=experimentalAccuracyThreshold16,
    x=listofNoiseValues,
    mode="markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="theta=16 (observed)"
)

trace1P = Scatter(
    y=predictedAccuracyThreshold8,
    x=listofNoiseValues,
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="theta=8 (predicted)"
)

trace2P = Scatter(
    y=predictedAccuracyThreshold12,
    x=listofNoiseValues,
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="theta=12 (predicted)"
)

trace3P = Scatter(
    y=predictedAccuracyThreshold16,
    x=listofNoiseValues,
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="theta=16 (predicted)"
)

data = Data([trace1P, trace2P, trace3P, trace1, trace2, trace3])

layout = Layout(
    title='',
    showlegend=False,
    autosize=False,
    width=855,
    height=700,
    xaxis=XAxis(
        title='Noise (percent of a)',
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
        dtick=0.1,
        showline=True,
        range=[0,1],
    ),
    yaxis=YAxis(
        title='Sequence accuracy',
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
            x=0.7,
            y=0.5,
            xref='x',
            yref='paper',
            text='$\\theta = 8$',
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
            x=0.5,
            y=0.5,
            xref='x',
            yref='paper',
            text='$\\theta = 12$',
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
            x=0.3,
            y=0.5,
            xref='x',
            yref='paper',
            text='$\\theta = 16$',
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
py.image.save_as(figure, 'images/effect_of_noise.png', scale=4)
