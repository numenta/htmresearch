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

plotlyUser = os.environ['PLOTLY_USERNAME']
plotlyAPIKey = os.environ['PLOTLY_API_KEY']


py.sign_in(plotlyUser, plotlyAPIKey)


# Calculated error values

# a=32 cells active out of n=2000 cells, theta=s/2
# Experimental errors
e1 = [500000./500000.0, 492576./950000.0, 33719./950000.0, 1746./950000.0, 102./950000.0, 3./950000.0, 0./950000.0, 0./950000.0, 0./950000.0, 0./950000.0, 0./950000.0, 0./950000.0, 0./50000.0, 0./50000.0, 0./50000.0]

# Calculated errors
e1_predicted = [0.99999990152361075639022198316974800245420818812577,
0.51813137109180305726118510352509531120417071418090,
0.035411290387821781867207163534731358068238554453221,
0.0018083668616482497750458034346701431544077347966161,
0.000090350083470168777031476927790297933256294292133239,
0.0000044358011648312473147928173689651453155881868865031,
0.00000021272534956876201761188450092912783915787645824874,
0.0000000099177126256727305872046099904086881903140731429689,
0.00000000044782989067061178152535857197602281554508779520284,
0.000000000019521706345883651604482642558208071830149987117788,
0.00000000000081909490610204585557476856526522907800057698347313,
0.000000000000032984853013662987415919466274290323584151046381248,
0.0000000000000012711756860457033041174625742822183239717225807028,
4.6742437724025379386377719296593623733041620932759e-17,
1.6347861202131285191181544405121659186831350579456e-18]

# a=32 cells active out of n=4000 cells, theta=s/2
# Experimental errors
e2 = [49981./50000, 8320./50000, 242./50000, 6./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000]

# Calculated errors
e2_predicted = [0.99967548539491804549604445080339348274243435001951,
0.16825062427388091352658412756559058920741692890377,
0.0045675055861596358553119002708296154187827790235307,
0.00011553688669298503057324139659083422324789835939622,
0.0000028978732482057440425466122781632397087265650870113,
0.000000071429891869813826909055769162740277746859826575744,
0.0000000017186903906149779115606153683439206277747800426177,
0.000000000040174410174347139046862766402045196041579480879900,
0.00000000000090885347634520495883315778975237305969411259760974,
0.000000000000019834493141457013169681439224687233922528799903677,
4.1632962872798232823106859381870135085034012342255e-16,
8.3809398111141169012851901965840245891317311686247e-18,
1.6133648609763571509806334190384448434972597914095e-19,
2.9611452353816206805968367900306998147789767804570e-21,
5.1653767067526762179637589002999159250772155276890e-23]

# a=512 cells active out of n=16000 cells, theta=s/2
# Experimental errors
e3 = [550000./550000.0,
521054./550000.0,
143674./550000.0,
17612./550000.0,
1972./550000.0,
233./550000.0,
29./550000.0,
2./550000.0,
1./600000.0,
0./100000.0,
0./100000.0,
0./100000.0,
0./100000.0,
0./100000.0,
0./100000.0]

# Calculated errors
e3_predicted = [0.999999999999992,
0.947454701131967,
0.261531393413475,
0.0322120516830234,
0.00362288342334083,
0.000408837119401984,
4.65763090697902e-5,
5.34598760215665e-6,
6.16972560707474e-7,
7.14887012920329e-8,
8.30757684866512e-9,
9.67448343658361e-10,
1.12854170453147e-10,
1.31561428418081e-11,
1.55431223447522e-12]

# a=4000 cells active out of n=16000 cells, theta=s/2
# Experimental errors
e4 = [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.998, 0.99, 0.988, 0.914, 0.812, 0.736]

# Calculated errors
e4_predicted = [1.00000000000000,
1.00000000000000,
1.00000000000000,
1.00000000000000,
1.00000000000000,
0.999999999999261,
0.999999996519808,
0.999998899282644,
0.999941045974400,
0.999045191721944,
0.993186460193815,
0.972473867256443,
0.925372946571690,
0.847361335075154,
0.744475803717050]

# X value range
synapses = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

trace1 = Scatter(
    y=e1,
    x=synapses,
    mode = "markers",
    marker=Marker(
      symbol="octagon",
      size=10,
      color="rgb(0, 0, 0)",
    ),
    name="a=32 n=2000"
)

trace1_predicted = Scatter(
    y=e1_predicted,
    x=synapses,
    mode = "lines",
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
    mode = "markers",
    marker=Marker(
      symbol="octagon",
      size=10,
      color="rgb(0, 0, 0)",
    ),
    name="a=32 n=4000"
)

trace2_predicted = Scatter(
    y=e2_predicted,
    x=synapses,
    mode = "lines",
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
    mode="markers",
    marker=Marker(
      symbol="octagon",
      size=10,
      color="rgb(0, 0, 0)",
    ),
    name="a=512 n=16000"
)

trace3_predicted = Scatter(
    y=e3_predicted,
    x=synapses,
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=512 n=16000"
)

trace4 = Scatter(
    y=e4,
    x=synapses,
    marker=Marker(
      symbol="octagon",
      size=10,
      color="rgb(0, 0,0 )",
    ),
    mode="markers",
    name="a=4000 n=16000"
)

trace4_predicted = Scatter(
    y=e4_predicted,
    x=synapses,
    mode="lines",
    line=Line(
        color='rgb(0, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=4000 n=16000"
)

data = Data([trace1, trace1_predicted,trace2, trace2_predicted, trace3, trace3_predicted, trace4, trace4_predicted])

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
            size=24,
            color='rgb(0, 0, 0)',
        ),
        tickfont=Font(
            family='',
            size=18,
            color='rgb(0, 0, 0)',
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
            size=24,
            color='rgb(0, 0, 0)',
        ),
        tickfont=Font(
            family='',
            size=18,
            color='rgb(0, 0, 0)',
        ),
        showline=True,
    ),
    annotations=Annotations([
      Annotation(
        # arrow -21, 60
            x=21.64,
            y=0.32,
            xref='x',
            yref='paper',
            text='$a = 32, n=4000$',
            showarrow=True,
            ax=-21,
            ay=60,
            font=Font(
                family='',
                size=24,
                color='rgba(0, 0, 0, 0)',
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
            y=0.51,
            xref='x',
            yref='paper',
            text='$a = 32, n=2000$',
            showarrow=True,
            ax = -89,
            ay = 72,
            font=Font(
                family='',
                size=24,
                color='rgba(0, 0, 0, 0)'
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
        # Arrow 42, -46
            x=19.44,
            y=0.67,
            xref='x',
            yref='paper',
            text='$a = 512, n=16000$',
            showarrow=True,
            ax = 42,
            ay=-46,
            font=Font(
                family='',
                size=24,
                color='rgba(0, 0, 0, 0)',
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
            y=0.95,
            xref='x',
            yref='paper',
            text='$a = 4000, n=16000$',
            showarrow=True,
            ax = 0,
            ay=-30,
            font=Font(
                family='',
                size=24,
                color='rgba(0, 0, 0, 0)',
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
py.image.save_as(figure, 'images/effect_of_s.pdf', scale=4)
