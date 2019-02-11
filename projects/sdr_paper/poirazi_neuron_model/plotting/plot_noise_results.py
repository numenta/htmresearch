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
experimentalErrorsThreshold8 =  [0,
0,
0,
0,
0./100000,
0./100000,
2./100000,
63./100000,
365./100000,
1765./100000,
7222./100000,
19271./100000,
44132./100000,
69847./100000]

theoreticalErrorsThreshold8 = [0, 0, 0, 3.11341516283240e-16,
2.23243424464799e-12, 7.90168637530306e-10, 1.20695707971664e-7,
3.45576265561118e-6, 8.08202472708491e-5, 0.000735429456875121,
0.00464043435771348, 0.0268657157114204, 0.0896352007201254, 0.263952754229579,
0.508714577385333, 0.770861966941236]


# a=128 cells active, s=30 synapses on segment
experimentalErrorsThreshold12 = [0./100000,
0./100000,
0./100000,
0./100000,
19./100000,
225./100000,
1306./100000,
6385./100000,
17036./100000,
35247./100000,
61938./100000,
81887./100000,
95315./100000,
99172./100000]

theoreticalErrorsThreshold12 = [0, 0, 2.48810797387309e-15,
7.92695349343630e-10, 2.16302525195240e-7, 1.09248135880715e-5,
0.000314435369055385, 0.00279559866084888, 0.0198782675563797,
0.0716985160403564, 0.190430462690358, 0.426525969583828, 0.664766152465367,
0.880922510721824, 0.970339402698393, 0.996376835285247]


# a=128 cells active, s=30 synapses on segment
experimentalErrorsThreshold16 =  [0./100000,
1./100000,
60./100000,
552./100000,
3195./100000,
12934./100000,
29597./100000,
55471./100000,
76287./100000,
90323./100000,
97765./100000,
99581./100000,
99972./100000,
100000./100000]


theoreticalErrorsThreshold16 = [0, 0, 2.65549705827547e-8, 1.97277260559420e-5,
0.000590078912236923, 0.00627504390204146, 0.0434711883681422,
0.139067254386793, 0.351117043857492, 0.582501773030327, 0.788076297517739,
0.933878292787173, 0.983735005386502, 0.998319255844748, 0.999889798557155,
0.999997748076386]


listofNoiseValues = [0.046875, 0.1015625, 0.1484375, 0.203125, 0.25, 0.296875,
0.3515625, 0.3984375, 0.453125, 0.5, 0.546875, 0.6015625, 0.6484375, 0.703125,
0.75, 0.796875]

trace8 = Scatter(
    y=experimentalErrorsThreshold8,
    x=listofNoiseValues[2:],
    mode="markers",
    marker=Marker(
      symbol="triangle",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="theta=8 (new observed)"
)
trace9 = Scatter(
    y=experimentalErrorsThreshold12,
    x=listofNoiseValues[2:],
    mode="markers",
    marker=Marker(
      symbol="triangle",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="theta=12 (new observed)"
)
trace10 = Scatter(
    y=experimentalErrorsThreshold16,
    x=listofNoiseValues[2:],
    mode="markers",
    marker=Marker(
      symbol="triangle",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="theta=16 (new observed)"
)

trace2 = Scatter(
    y=theoreticalErrorsThreshold8[3:],
    x=listofNoiseValues[3:],
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=2,
        dash='dot',
        shape='spline',
    ),
    name="theta=8 (predicted)"
)

trace4 = Scatter(
    y=theoreticalErrorsThreshold12[2:],
    x=listofNoiseValues[2:],
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=2,
        dash='dot',
        shape='spline',
    ),
    name="theta=12 (predicted)"
)

trace6 = Scatter(
    y=theoreticalErrorsThreshold16[2:],
    x=listofNoiseValues[2:],
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=2,
        dash='dot',
        shape='spline',
    ),
    name="theta=16 (predicted)"
)

data = Data([trace2, trace4, trace6, trace8, trace9, trace10])

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
        title='Frequency of false negatives',
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
            x=0.583,
            y=0.777,
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
            x=0.0993,
            y=0.2398,
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
            x=0.156,
            y=0.753,
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
plot_url = py.plot(fig, auto_open=False)
print "url=",plot_url
figure = py.get_figure(plot_url)
py.image.save_as(figure, 'images/effect_of_noise.pdf', scale=1)
