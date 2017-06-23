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
import numpy

plotlyUser = os.environ['PLOTLY_USERNAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']


py.sign_in(plotlyUser, plotlyAPIKey)


# experimental error values
errorsA10 = [0.1,
0.167739329962,
0.394798532689,
0.533994453588,
0.669309596322,
0.804351939058,
0.934421838533,
0.995305263158,
0.999884210526,
0.999894736842,
0.999947368421,
0.999947368421,
1.0,
0.999947368421,
0.999947368421,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0]

errorsA20 = [0.2,
0.134268022586,
0.201263157895,
0.268421052632,
0.335526315789,
0.402631578947,
0.469736842105,
0.536842105263,
0.603947368421,
0.671052631579,
0.738157894737,
0.805263157895,
0.872121052632,
0.934568421053,
0.978486842105,
0.996123684211,
0.999402631579,
0.999981578947,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0]

errorsA40 = [0.4,
0.2,
0.133333333333,
0.122443673108,
0.167584161291,
0.201268408827,
0.234847513921,
0.268413671374,
0.301971116816,
0.335525096277,
0.369078947368,
0.402631578947,
0.436184210526,
0.469736842105,
0.503289473684,
0.536842105263,
0.570394736842,
0.603947368421,
0.6375,
0.671052631579,
0.704605263158,
0.738157894737,
0.771710526316,
0.805263157895,
0.838801315789,
0.872088157895,
0.904684210526,
0.934184210526,
0.959640789474,
0.977884210526]



#errorsA10 = 1 - numpy.asarray(errorsA10)
#errorsA20 = 1 - numpy.asarray(errorsA20)
#errorsA40 = 1 - numpy.asarray(errorsA40)
listofNValues = range(100, 3100, 100)

trace1 = Scatter(
    y=errorsA10[3:],
    x=listofNValues[3:],
    mode = "lines+markers",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=10"
)

trace2 = Scatter(
    y=errorsA20[3:],
    x=listofNValues[3:],
    mode = "lines+markers",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=20"
)

trace3 = Scatter(
    y=errorsA40[3:],
    x=listofNValues[3:],
    mode = "lines+markers",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=40"
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
        dtick=500,
        showline=False,
        range=[0,3100],
    ),
    yaxis=YAxis(
        title='Sequence accuracy',
        type='linear',
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
            x=400,
            y=0.8,
            xref='x',
            yref='paper',
            text='a = 10',
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
            x=800,
            y=0.7,
            xref='x',
            yref='paper',
            text='a = 20',
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
            x=1600,
            y=0.6,
            xref='x',
            yref='paper',
            text='a = 40',
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
