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

# a=128 cells active, s=30 synapses on segment
experimentalErrorsThreshold8 = [1.0,
1.0,
1.0,
1.0,
0.999999353792,
0.999998129661,
1.0,
0.999965568561,
0.999321372128,
0.994241318088,
0.973815752686,
0.918610908135,
0.808554087142,
0.63753612854,
0.422839697448,
0.199114532601,
0.0387497874351,
0.00196960800926,
2.00186226158e-05,
-7.83166369663e-08]

experimentalErrorsThreshold82 = [1.0,
1.0,
1.0,
1.0,
0.999998716303,
0.999996328029,
1.0,
0.99993161104,
0.998652631579,
0.988597368421,
0.9487,
0.845202824134,
0.657226315789,
0.413329300385,
0.189123950119,
0.0545172366893,
0.00763796534018,
0.000352631578947,
3.94736842105e-06,
0.0]


# a=128 cells active, s=30 synapses on segment
experimentalErrorsThreshold12 = [1.0,
1.0,
1.0,
1.0,
1.0,
0.997938943255,
0.979989162525,
0.919609411662,
0.798247250587,
0.626131502313,
0.426165157198,
0.238029485615,
0.083054899818,
0.0136008996055,
0.000888738639035,
8.31930679339e-06,
0.0,
0.0,
0.0,
0.0]

experimentalErrorsThreshold122 = [1.0,
1.0,
1.0,
1.0,
1.0,
0.995910526316,
0.960664473684,
0.846871052632,
0.640214473684,
0.397481578947,
0.189097368421,
0.0672052631579,
0.0163776315789,
0.00227236842105,
0.000142105263158,
1.31578947368e-06,
0.0,
0.0,
0.0,
0.0]

# a=128 cells active, s=30 synapses on segment
experimentalErrorsThreshold16 = [1.0,
1.0,
1.0,
0.953432877996,
0.808744401517,
0.600566524795,
0.38757881838,
0.203285094949,
0.0703475125446,
0.0144245538297,
0.00182192818775,
9.98316815207e-05,
8.31930679339e-06,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0]

experimentalErrorsThreshold162 = [1.0,
1.0,
1.0,
0.909685526316,
0.656614473684,
0.365511842105,
0.156975,
0.0513763157895,
0.0129723684211,
0.00237368421053,
0.000288157894737,
1.57894736842e-05,
1.31578947368e-06,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0]

experimentalErrorsThreshold8 = numpy.asarray(experimentalErrorsThreshold82)
experimentalErrorsThreshold12 = numpy.asarray(experimentalErrorsThreshold122)
experimentalErrorsThreshold16 = numpy.asarray(experimentalErrorsThreshold162)

listofNoiseValues = [0, 0.046875, 0.1015625, 0.1484375, 0.203125, 0.25, 0.296875,
0.3515625, 0.3984375, 0.453125, 0.5, 0.546875, 0.6015625, 0.6484375, 0.703125,
0.75, 0.796875, 0.84375, 0.8984375, 0.9453125]

trace1 = Scatter(
    y=experimentalErrorsThreshold8,
    x=listofNoiseValues,
    mode="markers+lines",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="theta=8 (observed)"
)

trace2 = Scatter(
    y=experimentalErrorsThreshold12,
    x=listofNoiseValues,
    mode="markers+lines",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="theta=12 (observed)"
)

trace3 = Scatter(
    y=experimentalErrorsThreshold16,
    x=listofNoiseValues,
    mode="markers+lines",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="theta=16 (observed)"
)

data = Data([trace1, trace2, trace3])

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
            x=0.75,
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
            x=0.55,
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
            x=0.35,
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
