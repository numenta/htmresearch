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


# Calculated error values for the case where there is no subsampling, i.e.
# s=w

# s=w=16, dendritic threshold is theta=s/2
errorsW16 = [0.0624097796945172, 0.000457906987998791, 2.07105587746514e-5,
2.21255328532880e-6, 3.84968371385460e-7, 9.16381036769321e-8,
2.71336806884240e-8, 9.43446483548732e-9, 3.71050853337693e-9,
1.60872722198021e-9, 7.54845900719946e-10, 3.78120060364152e-10,
2.00115241484368e-10, 1.10991886700039e-10, 6.41015767420794e-11,
3.83492281041957e-11, 2.36650456857372e-11, 1.50102750498796e-11,
9.75698862526075e-12, 6.48337237434945e-12, 4.39455551087878e-12,
3.03289784920180e-12, 2.12782442409097e-12, 1.51545690731159e-12,
1.09433355322374e-12, 8.00359374030838e-13, 5.92288624619412e-13,
4.43122600903636e-13, 3.34907307572575e-13, 2.55527292248779e-13,
1.96694814204967e-13, 1.52668007297818e-13, 1.19421262119154e-13,
9.41006237732964e-14, 7.46615103801807e-14, 5.96246939284021e-14,
4.79100010952609e-14, 3.87217222906159e-14, 3.14688241060822e-14,
2.57088448799245e-14, 2.11080739825733e-14]

# w=32 s=32 synapses on segment, dendritic threshold is theta=s/2
errorsW32 = [0.999409841442712, 0.00842698413308383, 3.52017905373014e-5,
5.16312270867159e-7, 1.77041550313674e-8, 1.07987062475905e-9,
9.93722824578364e-11, 1.24312537377571e-11, 1.97220399882436e-12,
3.78032981837551e-13, 8.45300753938706e-14, 2.14793999132689e-14,
6.07928034208778e-15, 1.88662780247235e-15, 6.34005656392456e-16,
2.28389144529261e-16, 8.74627886027787e-17, 3.53620305536015e-17,
1.50071368493578e-17, 6.65225797421467e-18, 3.06707190448587e-18,
1.46550684870534e-18, 7.23425984778471e-19, 3.67916237853533e-19,
1.92310761162233e-19, 1.03093569801993e-19, 5.65731796224652e-20,
3.17254166762729e-20, 1.81537013628672e-20, 1.05850727484582e-20,
6.28146163015812e-21, 3.78951464000154e-21, 2.32179857163582e-21,
1.44339307819335e-21, 9.09702487894253e-22, 5.80811257411107e-22,
3.75392020615500e-22, 2.45452413730194e-22, 1.62263785643428e-22,
1.08394530336136e-22, 7.31310450203141e-23]

# s=w=64, dendritic threshold is theta=s/2
errorsW64 = [1.00, 0.999993040517434, 0.0808562612576227, 0.000203693205308507,
6.46805585794712e-7, 4.04458243823092e-9, 4.68881643886714e-11,
9.03958835496706e-13, 2.63759126198845e-14, 1.08156643649095e-15,
5.88516115305888e-17, 4.06356472120690e-18, 3.43712823304800e-19,
3.46257302371181e-20, 4.06089855342486e-21, 5.44198997189482e-22,
8.20515741424400e-23, 1.37399161642939e-23, 2.52752017452805e-24,
5.06017702469768e-25, 1.09374411944514e-25, 2.53473507918349e-26,
6.26027395128218e-27, 1.63908094545422e-27, 4.52824622545533e-28,
1.31459116364385e-28, 3.99566458017190e-29, 1.26736475436969e-29,
4.18265377855185e-30, 1.43248675178090e-30, 5.07901389628294e-31,
1.86027489966531e-31, 7.02470468882090e-32, 2.72993267902856e-32,
1.09002196212166e-32, 4.46502556156018e-33, 1.87377308425520e-33,
8.04564350494280e-34, 3.53056570116741e-34, 1.58159340452154e-34,
7.22560698773388e-35]


# w=n/2 cells active, w=n/2 synapses on segment, threshold is w/2
# Only the first 8 values are computed due to numerical precision issues
# The rest are duplicates, but should be approximately accurate
errorsWHalfOfN = [0.713930783595416, 0.579192330580665, 0.627949560779490,
0.556207787852021, 0.599827556589778, 0.545950866779313, 0.584632630831763,
0.539819496834182, 0.599827556589778, 0.545950866779313, 0.584632630831763,
0.556207787852021, 0.599827556589778, 0.545950866779313, 0.584632630831763,
0.556207787852021, 0.599827556589778, 0.545950866779313, 0.584632630831763,
0.539819496834182, 0.599827556589778, 0.545950866779313, 0.584632630831763,
0.556207787852021, 0.599827556589778, 0.545950866779313, 0.584632630831763,
0.556207787852021, 0.599827556589778, 0.545950866779313, 0.584632630831763,
0.539819496834182, 0.599827556589778, 0.545950866779313, 0.584632630831763,
0.556207787852021, 0.599827556589778, 0.545950866779313, 0.584632630831763,
0.599827556589778, 0.545950866779313]

listofNValues = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650,
700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350,
1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000,
2050]

trace1 = Scatter(
    y=errorsW64,
    x=listofNValues,
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="w=64"
)

trace2 = Scatter(
    y=errorsW32,
    x=listofNValues[1:],
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="w=32"
)

trace3 = Scatter(
    y=errorsW16,
    x=listofNValues[1:],
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="w=16"
)

trace4 = Scatter(
    y=errorsWHalfOfN,
    x=listofNValues[0:41],
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        dash='dash',
        shape='spline',
    ),
    name="w=0.5*n"
)

data = Data([trace1, trace2, trace3, trace4])

layout = Layout(
    title='',
    showlegend=False,
    autosize=False,
    width=855,
    height=700,
    xaxis=XAxis(
        title='SDR size (n)',
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
        dtick=200,
        showline=True,
        range=[0,2050],
    ),
    yaxis=YAxis(
        title='Probability of false positives',
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
            x=1730,
            y=0.1443,
            xref='x',
            yref='paper',
            text='$w = 64$',
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
            x=1730,
            y=0.4352,
            xref='x',
            yref='paper',
            text='$w = 32$',
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
            x=1730,
            y=0.6464,
            xref='x',
            yref='paper',
            text='$w = 16$',
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
            x=1730,
            y=0.933,
            xref='x',
            yref='paper',
            text='$w = \\frac{n}{2}$',
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
py.image.save_as(figure, 'images/effect_of_n_bami.pdf', scale=4)
