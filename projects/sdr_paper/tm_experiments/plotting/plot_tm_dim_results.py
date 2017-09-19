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
errorsA64 = [
0.213333333333,
0.16,
0.128,
0.109258927937,
0.540992088875,
0.976687550961,
0.99696026639,
0.998870703483,
0.999572931506,
0.99979538597,
0.999931198626,
0.999957109557,
0.999977327935,
0.999980566802,
0.999995951417,
0.999996761134,
0.99999757085,
0.999998380567,
0.999999190283,
1.0,
1.0,
0.999999190283,
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
1.0,
1.0]

theoreticalErrorsA64 = [
0.2133336330,
0.2432088366,
0.6044889510,
0.8816534815,
0.9678829427,
0.9904192682,
0.9967893123,
0.9988068068,
0.9995160176,
0.9997886271,
0.9999016873,
0.9999517284,
0.9999751567,
0.9999866758,
0.9999925887,
0.9999957417,
0.9999974813,
0.9999984707,
0.9999990492,
0.9999993960,
0.9999996087,
0.9999997419,
0.9999998269,
0.9999998821,
0.9999999186,
0.9999999430,
0.9999999596,
0.9999999710,
0.9999999790,
0.9999999846,
0.9999999887,
0.9999999915,
0.9999999936,
0.9999999952,
0.9999999963,
0.9999999972,
0.9999999978,
0.9999999983]


errorsA128 = [
0.426666666667,
0.32,
0.256,
0.213333333333,
0.182857142857,
0.16,
0.142222222222,
0.128,
0.116363636364,
0.107573073867,
0.165657816584,
0.432003745025,
0.81511543094,
0.953340076584,
0.99278275043,
0.995851676061,
0.997509372576,
0.998453538851,
0.999027416518,
0.999352075788,
0.999583629916,
0.999741807112,
0.999815208863,
0.999877211813,
0.99990331105,
0.999929416565,
0.999957568339,
0.999966136271,
0.999975934469,
0.999980824153,
0.999992656059,
0.999985312118,
0.999991024072,
0.999994288046,
0.999995512036,
0.99999755202,
0.999997144023,
0.99999877601]



theoreticalErrorsA128 = [
0.4266666667,
0.3200000000,
0.2560000000,
0.2133333597,
0.1859625575,
0.2206797387,
0.3533451947,
0.5472215973,
0.7278783408,
0.8507876739,
0.9207610294,
0.9577535390,
0.9770097587,
0.9871530152,
0.9926199755,
0.9956456398,
0.9973655923,
0.9983687205,
0.9989680252,
0.9993341645,
0.9995625291,
0.9997077134,
0.9998016645,
0.9998634680,
0.9999047488,
0.9999327161,
0.9999519161,
0.9999652616,
0.9999746458,
0.9999813169,
0.9999861080,
0.9999895823,
0.9999921247,
0.9999940013,
0.9999953977,
0.9999964449,
0.9999972359,
0.9999978376]


errorsA256 = [
0.853333333333,
0.64,
0.512,
0.426666666667,
0.365714285714,
0.32,
0.284444444444,
0.256,
0.232727272727,
0.213333333333,
0.196923076923,
0.182857142857,
0.170666666667,
0.16,
0.150588235294,
0.142222222222,
0.134736842105,
0.128,
0.121904761905,
0.116363636364,
0.111304347826,
0.106705710356,
0.10982613374,
0.139173704283,
0.222094765216,
0.37034715943,
0.551080386976,
0.730062493563,
0.912104778409,
0.948031898087,
0.974994299269,
0.988034361588,
0.990142928566,
0.995151107508,
0.996216040337,
0.997128197316,
0.997714802774,
0.998214039825]

theoreticalErrorsA256 = [
0.8533333333,
0.6400000000,
0.5120000000,
0.4266666667,
0.3657142857,
0.3200000000,
0.2844444444,
0.2560000000,
0.2327272727,
0.2133333406,
0.1969607947,
0.1849783758,
0.1868214993,
0.2115595973,
0.2609085096,
0.3328302851,
0.4220391337,
0.5199109607,
0.6164749974,
0.7034726858,
0.7763042934,
0.8339973398,
0.8779612893,
0.9106304006,
0.9345488681,
0.9519315819,
0.9645361372,
0.9736870465,
0.9803536777,
0.9852342625,
0.9888278913,
0.9914904453,
0.9934759273,
0.9949661871,
0.9960919987,
0.9969479054,
0.9976026541,
0.9981065378]

listofNValues = range(300, 4100, 100)

trace1 = Scatter(
    y=errorsA64[3:],
    x=listofNValues[3:],
    mode = "lines+markers",
    name="a=64"
)

trace2 = Scatter(
    y=errorsA128[9:],
    x=listofNValues[9:],
    mode = "lines+markers",
    name="a=128"
)

trace3 = Scatter(
    y=errorsA256[22:],
    x=listofNValues[22:],
    mode = "lines+markers",
    name="a=256"
)

trace1t = Scatter(
    y=theoreticalErrorsA64,
    x=listofNValues,
    mode = "lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=64"
)

trace2t = Scatter(
    y=theoreticalErrorsA128,
    x=listofNValues,
    mode = "lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=128"
)

trace3t = Scatter(
    y=theoreticalErrorsA256,
    x=listofNValues,
    mode = "lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=256"
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
        range=[500,3500],
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
            x=950,
            y=0.8,
            xref='x',
            yref='paper',
            text='a = 64',
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
            x=1700,
            y=0.7,
            xref='x',
            yref='paper',
            text='a = 128',
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
            x=3200,
            y=0.6,
            xref='x',
            yref='paper',
            text='a = 256',
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
