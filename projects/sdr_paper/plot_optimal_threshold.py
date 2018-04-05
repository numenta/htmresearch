# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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
# error rates as a function of theta.  I'm sorry this code is so ugly.

import plotly.plotly as py
from plotly.graph_objs import *
import os

plotlyUser = os.environ['PLOTLY_USERNAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']


py.sign_in(plotlyUser, plotlyAPIKey)

# theta	avgError	stdev	min	max	median
# 3	0.036662656	0.045076188	0.000126298	0.189184284	0.018156949
# 4	0.008381634	0.013555152	2.53E-06	0.062735256	0.002227752
# 5	0.001667118	0.00333653	3.73E-08	0.016796138	0.000210173
# 6	0.00029138	0.000692739	4.21E-10	0.003730784	1.58E-05
# 7	4.51E-05	0.000123628	3.72E-12	0.000702096	9.60E-07
# 8	6.22E-06	1.92E-05	2.61E-14	0.000113791	4.83E-08
# 9	7.68E-07	2.63E-06	1.47E-16	1.61E-05	2.04E-09
# 10	8.55E-08	3.20E-07	6.67E-19	2.01E-06	7.40E-11
# 11	8.60E-09	3.47E-08	2.44E-21	2.22E-07	2.49E-12
# 12	7.86E-10	3.38E-09	7.17E-24	2.21E-08	7.63E-14
# 13	6.54E-11	2.98E-10	1.69E-26	1.97E-09	2.05E-15
# 14	4.98E-12	2.39E-11	3.13E-29	1.60E-10	4.88E-17
# 15	3.47E-13	1.74E-12	4.54E-32	1.17E-11	1.03E-18
# 16	2.22E-14	1.15E-13	4.99E-35	7.88E-13	1.95E-20
# 17	1.31E-15	7.02E-15	4.01E-38	4.83E-14	3.28E-22
# 18	7.08E-17	3.92E-16	2.22E-41	2.72E-15	4.90E-24
# 19	3.54E-18	2.01E-17	7.50E-45	1.41E-16	5.32E-26
# 20	1.64E-19	9.51E-19	1.17E-48	6.69E-18	4.11E-28
# 21	7.00E-21	4.14E-20	0	2.94E-19	2.68E-30
# 22	2.77E-22	1.67E-21	0	1.19E-20	2.91E-32
# 23	1.01E-23	6.19E-23	0	4.45E-22	2.90E-34
# 24	3.43E-25	2.13E-24	0	1.54E-23	2.60E-36

# Median error values
errors = [0.018156949,0.002227752,0.000210173,1.58e-05,9.60e-07,4.83e-08,
          2.04e-09,7.40e-11,2.49e-12,7.63e-14,2.05e-15,4.88e-17,1.03e-18,
          1.95e-20,3.28e-22,4.90e-24,5.32e-26,4.11e-28,2.68e-30,2.91e-32,
          2.90e-34,2.60e-36]

# Compute log error, log(error - 2*stdev), log(error + 2*stdev)
lnerror= []
lbound = []
ubound = []
for i,e in enumerate(errors):
  lnerror.append(e)
  ubound.append(1e-9)
  lbound.append(errors[-1])

# trace1 = Scatter(
#     y=ubound,
#     x=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
#     line=Line(
#         color='rgb(239, 0, 22)',
#         width=3,
#         dash='dash'
#     ),
#     showlegend=False
# )
trace2_1 = Scatter(
    y=lnerror[0:7],
    x=[3,4,5,6,7,8,9,10],
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="error"
)
trace2_2 = Scatter(
    y=lnerror[6:],
    x=[9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="error"
)
trace3 = Scatter(
    y=lbound[6:],
    x=[9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    fill='tonexty',
    line=Line(
        color='rgb(255, 255, 255)',
        width=1,
    ),
    fillcolor='rgba(217, 217, 217, 0.5)',
    showlegend=False
)

data = Data([trace2_1, trace2_2, trace3])

layout = Layout(
    title='',
    showlegend=False,
    autosize=False,
    width=855,
    height=700,
    xaxis=XAxis(
        title='$\\text{No. of synapses required for dendritic spike } ('
              '\\theta)$',
        titlefont=Font(
            family='Arial',
            size=24,
            color='rgb(0, 0, 0)',
        ),
        tickfont=Font(
            family='Arial',
            size=24,
            color='rgb(0, 0, 0)',
        ),
        showline=True,
    ),
    yaxis=YAxis(
        title='Probability of false positives',
        type='log',
        exponentformat='power',
        autorange=True,
        titlefont=Font(
            family='Arial',
            size=24,
            color='rgb(0, 0, 0)',
        ),
        tickfont=Font(
            family='Arial',
            size=16,
            color='rgb(0, 0, 0)',
        ),
        # showticklabels=False,
        showline=True,
    ),
    annotations=Annotations([
      Annotation(
            x=14.776699029126213,
            y=0.5538461538461539,
            xref='x',
            yref='paper',
            text='Median false positive error',
            showarrow=True,
            font=Font(
                family='Arial',
                size=24,
                color='rgb(0, 0, 0)',
            ),
            xanchor='auto',
            yanchor='auto',
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=0,
            arrowcolor='',
            ax=107,
            ay=-46.171875,
            textangle=0,
            bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
    Annotation(
          x=9.79611650485437,
          y=0.46153846153846156,
          xref='x',
          yref='paper',
          text='$\\text{Error rate} \leq 10^{-9}$',
          showarrow=True,
          font=Font(
              family='Arial',
              size=24,
              color='rgb(0, 0, 0)',
          ),
          xanchor='auto',
          yanchor='auto',
          align='center',
          arrowhead=2,
          arrowsize=1,
          arrowwidth=0,
          arrowcolor='',
          ax=-109,
          ay=63.828125,
          textangle=0,
          bordercolor='',
          borderwidth=1,
          borderpad=1,
          bgcolor='rgba(0,0,0,0)',
          opacity=1
      )
    ]),
)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig, auto_open=False)
print "url=",plot_url
figure = py.get_figure(plot_url)
py.image.save_as(figure, 'images/optimal_threshold.pdf', scale=2)