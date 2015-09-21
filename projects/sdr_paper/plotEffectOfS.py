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
# error rates as a function of N, the dimensionality of the vectors.  I'm sorry
# this code is so ugly.

import plotly.plotly as py
from plotly.graph_objs import *
import os

plotlyUser = os.environ['PLOTLY_USER_NAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']


py.sign_in(plotlyUser, plotlyAPIKey)


# Calculated error values
errors = [0.0975118779694924, 0.0139611629239578, 0.00220175230959234,
0.000362364391941437, 6.10197168557792e-5, 1.04143682425737e-5,
1.79201723563779e-6, 3.09874827644910e-7, 5.37318556612265e-8,
9.32884283256879e-9, 1.61994639311239e-9, 2.81121608709614e-10,
4.87228709396082e-11, 8.42942705148545e-12, 1.45516736670963e-12]

synapses = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

# # Compute log error, log(error - 2*stdev), log(error + 2*stdev)
# lnerror= []
# for i,e in enumerate(errors):
#   lnerror.append(e)
#

trace2_1 = Scatter(
    y=errors,
    x=synapses,
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="error"
)

data = Data([trace2_1])

layout = Layout(
    title='',
    showlegend=False,
    autosize=False,
    width=855,
    height=700,
    xaxis=XAxis(
        title='Number of synapses on segment',
        titlefont=Font(
            family='',
            size=16,
            color=''
        ),
        tickfont=Font(
            family='',
            size=16,
            color=''
        ),
        exponentformat="none",
        showline=True,
        range=[0,30],
    ),
    yaxis=YAxis(
        title='Probability of false positives',
        type='log',
        exponentformat='power',
        autorange=True,
        titlefont=Font(
            family='',
            size=18,
            color=''
        ),
        tickfont=Font(
            family='',
            size=12,
            color=''
        ),
        showline=True,
    ),
)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
print "url=",plot_url
figure = py.get_figure(plot_url)
py.image.save_as(figure, 'figure.png')