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
from plotly.graph_objs import *
import os

plotlyUser = os.environ['PLOTLY_USERNAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']


py.sign_in(plotlyUser, plotlyAPIKey)


# Calculated error values

# w=200 cells active out of population of n=4000 cells
errors = [0.0975118779694924, 0.0139611629239578, 0.00220175230959234,
0.000362364391941437, 6.10197168557792e-5, 1.04143682425737e-5,
1.79201723563779e-6, 3.09874827644910e-7, 5.37318556612265e-8,
9.32884283256879e-9, 1.61994639311239e-9, 2.81121608709614e-10,
4.87228709396082e-11, 8.42942705148545e-12, 1.45516736670963e-12]

# w=128 cells active out of n=4000 cells, theta=s/2
errorsA128 = [0.0629837459364841, 0.00584430512527604, 0.000596596501225661,
6.34149149902892e-5, 6.87897949491240e-6, 7.54205223721698e-7,
8.31279723703759e-8, 9.18014626543976e-9, 1.01352464978094e-9,
1.11692707533825e-10, 1.22722470782534e-11, 1.34323936472554e-12,
1.46357551285682e-13, 1.58660559533635e-14, 1.71047348036875e-15]


# w=256 cells active out of n=8000 cells, theta=s/2
errorsA256 = [0.0629798724840605, 0.00586466274476339, 0.000602973198348946,
6.47893770382337e-5, 7.13081732853977e-6, 7.96226067701095e-7,
8.97171982962439e-8, 1.01679133638258e-8, 1.15655095900038e-9,
1.31832019128216e-10, 1.50427273255654e-11, 1.71682860211354e-12,
1.95861312395311e-13, 2.23242197462621e-14, 2.54118657161951e-15]

# w=512 cells active out of n=16000 cells, theta=s/2
errorsA512 = [0.0629779361210076, 0.00587483456448912, 0.000606169580170940,
6.54824106515384e-5, 7.25889757236790e-6, 8.17841097774893e-7,
9.31547832439269e-8, 1.06920037239144e-8, 1.23394545221835e-9,
1.42977409702263e-10, 1.66151854939354e-11, 1.93490570019431e-12,
2.25664581841877e-13, 2.63454758717112e-14, 3.07765573247024e-15]

# w=32 cells active out of n=2000 cells, theta=s/2
e1 = [0.0317518759379690, 0.00145910198652343, 7.21043526586462e-5,
3.62000130960674e-6, 1.80708314241898e-7, 8.87162196669982e-9,
4.25450744299101e-10, 1.98354253495098e-11, 8.95659781541374e-13,
3.90434126921476e-14, 1.63818981220476e-15, 6.59697060273271e-17,
2.54235137209141e-18, 9.34848754480508e-20, 3.26957224042626e-21]

# w=32 cells active out of n=4000 cells, theta=s/2
e2 = [0.0159379844961240, 0.000368380360563884, 9.15589510774173e-6,
2.31087096485802e-7, 5.79575487730169e-9, 1.42859788831653e-10,
3.43738078417794e-12, 8.03488203503050e-14, 1.81770695269123e-15,
3.96689862829144e-17, 8.32659257455965e-19, 1.67618796222282e-20,
3.22672972195271e-22, 5.92229047076324e-24, 1.03307534135054e-25]

# w=256 cells active out of n=16000 cells, theta=s/2
e3 = [0.0317449840615038, 0.00149789666286458, 7.81322668823010e-5,
4.26087025535749e-6, 2.38035717510188e-7, 1.34908860116719e-8,
7.71519657301477e-10, 4.43732367140564e-11, 2.56102362062069e-12,
1.48103355887014e-13, 8.57230129616979e-15, 4.96196289826140e-16,
2.87050187411583e-17, 1.65879779753702e-18, 9.57163117212011e-20]

# w=512 cells active out of n=16000 cells, theta=s/2
e4 = [0.0629779361210076, 0.00587483456448912, 0.000606169580170940,
6.54824106515384e-5, 7.25889757236790e-6, 8.17841097774893e-7,
9.31547832439269e-8, 1.06920037239144e-8, 1.23394545221835e-9,
1.42977409702263e-10, 1.66151854939354e-11, 1.93490570019431e-12,
2.25664581841877e-13, 2.63454758717112e-14, 3.07765573247024e-15]

# w=4000 cells active out of n=16000 cells, theta=s/2
e5 = [0.437511719482468, 0.261705563155493, 0.169396504590094,
0.113761218620382, 0.0780630164280560, 0.0543344667231431, 0.0382034690567412,
0.0270660645556014, 0.0192890442507909, 0.0138117305771507, 0.00992807501953914,
0.00715948677425288, 0.00517705699047973, 0.00375230607503995,
0.00272515634577904]

synapses = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

trace1 = Scatter(
    y=e1,
    x=synapses,
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=32 n=2000"
)

trace2 = Scatter(
    y=e2,
    x=synapses,
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=32 n=4000"
)

trace3 = Scatter(
    y=e3,
    x=synapses,
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=256 n=16000"
)

trace4 = Scatter(
    y=e4,
    x=synapses,
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=512 n=16000"
)

trace5 = Scatter(
    y=e5,
    x=synapses,
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        dash='dash',
        shape='spline',
    ),
    name="a=4000 n=16000"
)


data = Data([trace1, trace2, trace4, trace5])

layout = Layout(
    title='',
    showlegend=False,
    autosize=False,
    width=855,
    height=700,
    xaxis=XAxis(
        title='$\\text{Number of subsampled bits, } w_x$',
        titlefont=Font(
            family='',
            size=22,
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
            size=22,
            color=''
        ),
        ticks='outside',
        tickfont=Font(
            family='',
            size=12,
            color=''
        ),
        showline=True,
    ),
    annotations=Annotations([
      Annotation(
        # arrow -21, 60
            x=21.64,
            y=0.28,
            xref='x',
            yref='paper',
            text='$w_y = 32, n=4000$',
            showarrow=True,
            ax=-21,
            ay=60,
            font=Font(
                family='',
                size=16,
                color=''
            ),
            align='right',
            textangle=0,
            bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
      Annotation(
        # Arrow -89, 72
            x=19.32,
            y=0.47,
            xref='x',
            yref='paper',
            text='$w_y = 32, n=2000$',
            showarrow=True,
            ax = -89,
            ay = 72,
            font=Font(
                family='',
                size=16,
                color=''
            ),
            align='right',
            textangle=0,
            bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
      # Annotation(
      #   # Arrow 58, -99
      #       x=24.66,
      #       y=0.41,
      #       xref='x',
      #       yref='paper',
      #       text='$a = 256, n=16000$',
      #       showarrow=True,
      #       ax = 58,
      #       ay = -99,
      #       font=Font(
      #           family='',
      #           size=16,
      #           color=''
      #       ),
      #       align='left',
      #       textangle=0,
      #       bordercolor='',
      #       borderwidth=1,
      #       borderpad=1,
      #       bgcolor='rgba(0, 0, 0, 0)',
      #       opacity=1
      #   ),
      Annotation(
        # Arrow 42, -46
            x=19.44,
            y=0.63,
            xref='x',
            yref='paper',
            text='$w_y = 512, n=16000$',
            showarrow=True,
            ax = 42,
            ay=-46,
            font=Font(
                family='',
                size=16,
                color=''
            ),
            align='left',
            textangle=0,
            bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
      Annotation(
            x=23.67,
            y=0.90,
            xref='x',
            yref='paper',
            text='$w_y = 4000, n=16000$',
            showarrow=True,
            ax = 0,
            ay=-30,
            font=Font(
                family='',
                size=16,
                color=''
            ),
            align='left',
            textangle=0,
            bordercolor='',
            borderwidth=1,
            borderpad=1,
            bgcolor='rgba(0, 0, 0, 0)',
            opacity=1
        ),
      ]),
    )

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
print "url=",plot_url
figure = py.get_figure(plot_url)
py.image.save_as(figure, 'images/effect_of_s_bami.pdf', scale=4)