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
experimental_overlap_2000_40 = 1 - numpy.asarray([0.01985,
0.039845,
0.0588375,
0.0777125,
0.0966275,
0.1140825,
0.132145,
0.149525,
0.1659375,
0.1830275,
0.19895,
0.2152975,
0.230135,
0.247375,
0.2599125,
0.2759075,
0.2910225,
0.3057575,
0.319655,
0.3317025,
0.34505,
0.3583175,
0.3713575,
0.3840075,
0.39714,
0.40927,
0.4208775,
0.43037,
0.44326,
0.4535525,
0.4646925,
0.47689,
0.485185,
0.4967075,
0.5081,
0.5171425,
0.5259075,
0.534675,
0.5467275,
0.55261,
0.56339,
0.572005,
0.57995,
0.5881675,
0.59737,
0.6052275,
0.61329,
0.62076,
0.628665,
0.6365225])

predicted_overlap_2000_40 = 1 - numpy.asarray([0.02000000000,
0.03960000000,
0.05880800000,
0.07763184000,
0.09607920320,
0.1141576191,
0.1318744668,
0.1492369774,
0.1662522379,
0.1829271931,
0.1992686493,
0.2152832763,
0.2309776107,
0.2463580585,
0.2614308974,
0.2762022794,
0.2906782338,
0.3048646691,
0.3187673758,
0.3323920282,
0.3457441877,
0.3588293039,
0.3716527178,
0.3842196635,
0.3965352702,
0.4086045648,
0.4204324735,
0.4320238240,
0.4433833476,
0.4545156806,
0.4654253670,
0.4761168597,
0.4865945225,
0.4968626320,
0.5069253794,
0.5167868718,
0.5264511344,
0.5359221117,
0.5452036694,
0.5542995960,
0.5632136041,
0.5719493321,
0.5805103454,
0.5889001385,
0.5971221357,
0.6051796930,
0.6130760992,
0.6208145772,
0.6283982856,
0.6358303199])


experimental_overlap_4000_40 = 1 - numpy.asarray([0.00989,
0.01959,
0.0299175,
0.038975,
0.0491425,
0.0582575,
0.068725,
0.07707,
0.0861,
0.0952225,
0.1043975,
0.1144475,
0.1217625,
0.13139,
0.14007,
0.1478475,
0.1577875,
0.1655725,
0.1741725,
0.1822025,
0.1884,
0.19814,
0.20723,
0.2150075,
0.2222525,
0.2298725,
0.2380625,
0.24561,
0.25228,
0.2592075,
0.2676725,
0.275025,
0.2850925,
0.2889875,
0.2964475,
0.3045125,
0.3095575,
0.3164275,
0.3240025,
0.3318925,
0.337145,
0.34453,
0.351175,
0.357335,
0.3636425,
0.3708825,
0.3751825,
0.3811675,
0.3870225,
0.3951375])


predicted_overlap_4000_40 = 1 - numpy.asarray([0.01000000000,
0.01990000000,
0.02970100000,
0.03940399000,
0.04900995010,
0.05851985060,
0.06793465209,
0.07725530557,
0.08648275251,
0.09561792499,
0.1046617457,
0.1136151283,
0.1224789770,
0.1312541872,
0.1399416454,
0.1485422289,
0.1570568066,
0.1654862385,
0.1738313762,
0.1820930624,
0.1902721318,
0.1983694105,
0.2063857164,
0.2143218592,
0.2221786406,
0.2299568542,
0.2376572857,
0.2452807128,
0.2528279057,
0.2602996266,
0.2676966303,
0.2750196640,
0.2822694674,
0.2894467727,
0.2965523050,
0.3035867819,
0.3105509141,
0.3174454050,
0.3242709509,
0.3310282414,
0.3377179590,
0.3443407794,
0.3508973716,
0.3573883979,
0.3638145139,
0.3701763688,
0.3764746051,
0.3827098591,
0.3888827605,
0.3949939329,
0.4010439935,
0.4070335536,
0.4129632181,
0.4188335859,
0.4246452500,
0.4303987975,
0.4360948095,
0.4417338615,
0.4473165228,
0.4528433576,
0.4583149240,
0.4637317748,
0.4690944570,
0.4744035125,
0.4796594774,
0.4848628826,
0.4900142538,
0.4951141112,
0.5001629701,
0.5051613404,
0.5101097270,
0.5150086297,
0.5198585434,
0.5246599580,
0.5294133584,
0.5341192248,
0.5387780326,
0.5433902523,
0.5479563497,
0.5524767862,
0.5569520184,
0.5613824982,
0.5657686732,
0.5701109865,
0.5744098766,
0.5786657778,
0.5828791201,
0.5870503289,
0.5911798256,
0.5952680273,
0.5993153470,
0.6033221936,
0.6072889716,
0.6112160819,
0.6151039211,
0.6189528819,
0.6227633531,
0.6265357195,
0.6302703624])

num_patterns_range = range(1, 51)

trace1 = Scatter(
    y=experimental_overlap_2000_40,
    x=num_patterns_range,
    mode = "markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="n=2000 (experimental)"
)

trace2 = Scatter(
    y=predicted_overlap_2000_40,
    x=num_patterns_range,
    mode = "lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="n=2000 (predicted)"
)
trace3 = Scatter(
    y=experimental_overlap_4000_40,
    x=num_patterns_range,
    mode = "markers",
    marker=Marker(
      symbol="octagon",
      size=12,
      color="rgb(0, 0, 0)",
    ),
    name="n=4000 (experimental)"
)
trace4 = Scatter(
    y=predicted_overlap_4000_40,
    x=num_patterns_range,
    mode = "lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="n=4000 (predicted)"
)

data = Data([trace2, trace1, trace4, trace3])

layout = Layout(
    title='',
    showlegend=False,
    autosize=False,
    width=855,
    height=700,
    xaxis=XAxis(
        title='Number of patterns',
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
        dtick=5,
        showline=True,
        range=[0, 60],
    ),
    yaxis=YAxis(
        title='Surprise',
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
                x=23,
                y=0.53,
                xref='x',
                yref='paper',
                text='$n = 2000$',
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
                x=32.76,
                y=0.667,
                xref='x',
                yref='paper',
                text='$n = 4000$',
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
            )
        ]),)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
print "url=",plot_url
figure = py.get_figure(plot_url)
py.image.save_as(figure, 'images/effect_of_n.png', scale=4)
