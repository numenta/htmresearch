# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
import plotly.graph_objs as go
import os

plotlyUser = os.environ['PLOTLY_USERNAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']


py.sign_in(plotlyUser, plotlyAPIKey)


# a=100 cells active out of n=2000 cells, theta=s/2
e6 = [0.0975237618809405, 0.0139034702742070, 0.00217374378984872,
0.000353090020219646, 5.84170413983881e-5, 9.75049857153054e-6,
1.63312727295096e-6, 2.73571449203843e-7, 4.57306938532383e-8,
7.61625039778459e-9, 1.26228575796699e-9, 2.07998415143951e-10,
3.40510511750091e-11, 5.53487063908265e-12, 8.92831862327263e-13]

# a=40 cells active out of n=2000 cells, theta=s/2
e8 = [0.0396098049024512, 0.00228221405278136, 0.000142324031139738,
9.08136231104620e-6, 5.80570382679830e-7, 3.68006347871095e-8,
2.29861511070827e-9, 1.40890947526859e-10, 8.44827103476593e-12,
4.94363120173214e-13, 2.81706539778811e-14, 1.56023004774561e-15,
8.38373779699067e-17, 4.36296052582060e-18, 2.19511505986332e-19]

# a=200 cells active out of n=2000 cells, theta=s/2
e9 = [0.190045022511256, 0.0521295878713446, 0.0156859149165168,
0.00491901819614129, 0.00157772648687802, 0.000512922385877911,
0.000168169308938023, 5.54323256302167e-5, 1.83317191611403e-5,
6.07350803856267e-6, 2.01379446070817e-6, 6.67705902339958e-7,
2.21249741008298e-7, 7.32311486638779e-8, 2.42020811438693e-8]

e10 = [0.640120060030015, 0.524886496144233, 0.455679956691754,
0.405797075497994, 0.366644926373627, 0.334392331818442, 0.306994047782892,
0.283224284072791, 0.262286447029172, 0.243630950057104, 0.226860784034208,
0.211678534357409, 0.197854739593335, 0.185208018007857, 0.173592054463543]

synapses = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]



trace6 = go.Scatter(
    y=e6,
    x=synapses,
    line=dict(
        color='rgb(0, 0, 0)',
        width=3,
        dash='dash',
        shape='spline',
    ),
    name="Sparsity = 1%"
)

trace8 = go.Scatter(
    y=e8,
    x=synapses,
    line=dict(
        color='rgb(0, 0, 0)',
        width=3,
        dash='dash',
        shape='spline',
    ),
    name="L6 size=4000, sparsity = 3%"
)

trace9 = go.Scatter(
    y=e9,
    x=synapses,
    line=dict(
        color='rgb(0, 0, 0)',
        width=3,
        dash='dash',
        shape='spline',
    ),
    name="L6 size=4000, sparsity = 3%"
)

trace10 = go.Scatter(
    y=e10,
    x=synapses,
    line=dict(
        color='rgb(0, 0, 0)',
        width=3,
        dash='dash',
        shape='spline',
    ),
    name="L6 size=4000, sparsity = 3%"
)

data = [trace6, trace8, trace9, trace10]

layout = go.Layout(
    title='',
    showlegend=False,
    autosize=False,
    width=855,
    height=700,
    xaxis=dict(
        title='No. of synapses on TRN dendritic segment',
        titlefont=dict(
            family='Arial',
            size=24,
            color='rgb(0, 0, 0)',
        ),
        tickfont=dict(
            family='Arial',
            size=18,
            color='rgb(0, 0, 0)',
        ),
        exponentformat="none",
        showline=True,
        range=[0,30],
    ),
    yaxis=dict(
        title='Probability of false positive error',
        type='log',
        exponentformat='power',
        autorange=True,
        titlefont=dict(
            family='Arial',
            size=24,
            color='rgb(0, 0, 0)',
        ),
        tickfont=dict(
            family='Arial',
            size=18,
            color='rgb(0, 0, 0)',
        ),
        showline=True,
    ),
    annotations=[
      dict(
        # arrow -21, 60
            x=21.64,
            y=0.28,
            xref='x',
            yref='paper',
            text='L6 sparsity=2%',
            showarrow=True,
            ax=-21,
            ay=60,
            # font=Font(
            #     family='',
            #     size=24,
            #     color='rgba(0, 0, 0, 0)',
            # ),
            align='right',
            textangle=0,
            # bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
      dict(
        # Arrow -89, 72
            x=22,
            y=0.56,
            xref='x',
            yref='paper',
            text='L6 sparsity=5%',
            showarrow=True,
            # ax = -89,
            # ay = 72,
            # font=Font(
            #     family='',
            #     size=24,
            #     color='rgba(0, 0, 0, 0)'
            # ),
            align='right',
            textangle=0,
            # bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
      dict(
        # Arrow 42, -46
            x=19,
            y=0.75,
            xref='x',
            yref='paper',
            text='L6 sparsity=10%',
            showarrow=True,
            ax = 42,
            ay=-46,
            # font=dict(
            #     family='Arial',
            #     size=24,
            #     color='rgba(0, 0, 0, 0)',
            # ),
            align='right',
            textangle=0,
            # bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
      ),
      dict(
        # Arrow 42, -46
        x=25,
        y=0.95,
        xref='x',
        yref='paper',
        text='L6 sparsity=40%',
        showarrow=True,
        # ax=42,
        # ay=-46,
        # font=dict(
        #     family='Arial',
        #     size=24,
        #     color='rgba(0, 0, 0, 0)',
        # ),
        align='right',
        textangle=0,
        # bordercolor='',
        borderwidth=1,
        borderpad=1,
        bgcolor='rgba(0, 0, 0, 0)',
        opacity=1
      )
    ],
  )
#
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, auto_open=False)
print "url=",plot_url
figure = py.get_figure(plot_url)
py.image.save_as(figure, 'images/effect_of_s_thalamus.pdf', scale=4)
