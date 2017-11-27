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
experimentalErrorsA64 = [ 1.09318E-03, 5.74000E-06, 1.10000E-07]

experimentalErrorsA64_poirazi_mel = [2736527./20000000, 621./10000000, 2./10000000, 0./10000000]


theoreticalErrorsA64 = [0.00109461662333690, 5.69571108769533e-6,
1.41253230930730e-7]


# a=128 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimentalErrorsA128 = [ 0.292048, 0.00737836, 0.00032014, 0.00002585,
0.00000295, 0.00000059, 0.00000013, 0.00000001, 0.00000001 ]

experimentalErrorsA128_poirazi_mel = [959346./2000000,
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

theoreticalErrorsA128_poirazi_mel = [
0.494226497771251,
0.336545078358176,
0.0315172776289802,
0.000859480824533425,
5.91397325008749e-5,
6.28687094584633e-6,
9.04180821116889e-7,
1.63365078351691e-7,
3.53297912640814e-8,
8.83820357552451e-9,
2.49387153179035e-9,
7.78644583624855e-10,
2.65012553737229e-10,
9.71628116464715e-11]

theoreticalErrorsA128 = [0.292078213737764, 0.00736788303358289,
0.000320106080889471, 2.50255519815378e-5, 2.99642102590114e-6,
4.89399786076359e-7, 1.00958512780931e-7, 2.49639031779358e-8,
7.13143762262004e-9]

# a=256 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimentalErrorsA256 = [
9.97368E-01, 6.29267E-01, 1.21048E-01, 1.93688E-02, 3.50879E-03, 7.49560E-04,
1.86590E-04, 5.33200E-05, 1.65000E-05, 5.58000E-06, 2.23000E-06, 9.30000E-07,
3.20000E-07, 2.70000E-07, 7.00000E-08, 4.00000E-08, 2.00000E-08
]

experimentalErrorsA256_poirazi_mel = [1001193./2000000,
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


theoreticalErrorsA256_poirazi_mel = [
0.5,
0.4903216662115622223336346073497079738271785250717125504215169949497802961343463923702215397563900091,
0.4660429024933562534895097402598522846282126783595503338575726881695410521741996601935625958422687864,
0.375845600482030,
0.215470686208248,
0.0442820703447854,
0.00511885431084768,
0.000664466647232933,
0.000117038891696029,
2.69196723399564e-5,
7.48278777837437e-6,
2.37417598572450e-6,
8.30763693093642e-7,
3.14063787448385e-7]

theoreticalErrorsA256 = [ 0.999997973443107, 0.629372754740777,
0.121087724790945, 0.0193597645959856, 0.00350549721741729,
0.000748965962032781, 0.000186510373919969, 5.30069204544174e-5,
1.68542688790000e-5, 5.89560747849969e-6, 2.23767020178735e-6,
9.11225564771580e-7, 3.94475072403605e-7, 1.80169987461924e-7,
8.62734957588259e-8, 4.30835081022293e-8, 2.23380881095835e-8]

listofNValues = [300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300,
2500, 2700, 2900, 3100, 3300, 3500]


trace1 = Scatter(
    y=experimentalErrorsA64,
    x=listofNValues[0:3],
    mode="markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="a=64 (observed)"
)

trace1_poirazi_mel = Scatter(
    y=experimentalErrorsA64_poirazi_mel,
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
    y=theoreticalErrorsA64,
    x=listofNValues[0:3],
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=2,
        dash='dot',
        shape='spline',
    ),
    name="a=64 (predicted)"
)


trace3 = Scatter(
    y=experimentalErrorsA128,
    x=listofNValues[0:9],
    mode="markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="a=128 (observed)"
)

trace3_poirazi_mel = Scatter(
    y=experimentalErrorsA128_poirazi_mel,
    x=listofNValues[0:9],
    mode="lines + markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="a=128 (observed)"
)

trace4 = Scatter(
    y=theoreticalErrorsA128,
    x=listofNValues[0:9],
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=2,
        dash='dot',
        shape='spline',
    ),
    name="a=128 (predicted)"
)
trace4_poirazi_mel = Scatter(
    y=theoreticalErrorsA128_poirazi_mel,
    x=listofNValues[0:9],
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=2,
        dash='dot',
        shape='spline',
    ),
    name="a=128 (predicted)"
)


trace5 = Scatter(
    y=experimentalErrorsA256,
    x=listofNValues,
    mode="markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="a=256 (observed)"
)

trace5_poirazi_mel = Scatter(
    y=experimentalErrorsA256_poirazi_mel,
    x=listofNValues,
    mode="lines+markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="a=256 (observed)"
)

trace6 = Scatter(
    y=theoreticalErrorsA256,
    x=listofNValues,
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=2,
        dash='dot',
        shape='spline',
    ),
    name="a=256 (predicted)"
)

trace6_poirazi_mel = Scatter(
    y=theoreticalErrorsA256_poirazi_mel,
    x=listofNValues,
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=2,
        dash='dot',
        shape='spline',
    ),
    name="a=256 (predicted)"
)

#data = Data([trace1, trace1_poirazi_mel, trace2, trace3, trace3_poirazi_mel, trace4, trace4_poirazi_mel, trace5, trace5_poirazi_mel, trace6, trace6_poirazi_mel])
data = Data([trace1_poirazi_mel, trace3_poirazi_mel, trace5_poirazi_mel])
layout = Layout(
    title='',
    showlegend=False,
    autosize=False,
    width=855,
    height=700,
    xaxis=XAxis(
        title='Cell population size (n)',
        titlefont=Font(
            family='Arial',
            size=26,
            color=''
        ),
        tickfont=Font(
            family='Arial',
            size=24,
            color=''
        ),
        exponentformat="none",
        dtick=800,
        showline=True,
        range=[0,4000],
    ),
    yaxis=YAxis(
        title='Binary classification error',
        type='log',
        exponentformat='power',
        autorange=True,
        ticks='outside',
        titlefont=Font(
            family='Arial',
            size=26,
            color=''
        ),
        tickfont=Font(
            family='Arial',
            size=16,
            color=''
        ),
        showticklabels=True,
        # ticksuffix="",
        showline=True,
    ),
    annotations=Annotations([
      Annotation(
            x=300,
            y=0.2085,
            xref='x',
            yref='paper',
            text='$a=64$',
            showarrow=False,
            font=Font(
                family='Arial',
                size=24,
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
            text='$a=128$',
            showarrow=False,
            font=Font(
                family='Arial',
                size=24,
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
            text='$a=256$',
            showarrow=False,
            font=Font(
                family='Arial',
                size=24,
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
plot_url = py.plot(fig, auto_open=True)
print "url=",plot_url
figure = py.get_figure(plot_url)
# py.image.save_as(figure, 'images/effect_of_n.png', scale=4)
py.image.save_as(figure, 'images/effect_of_n.pdf', scale=1)
