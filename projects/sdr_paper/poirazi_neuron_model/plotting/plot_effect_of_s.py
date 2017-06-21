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

# a=200 cells active out of population of n=4000 cells
errors = [0.0975118779694924, 0.0139611629239578, 0.00220175230959234,
0.000362364391941437, 6.10197168557792e-5, 1.04143682425737e-5,
1.79201723563779e-6, 3.09874827644910e-7, 5.37318556612265e-8,
9.32884283256879e-9, 1.61994639311239e-9, 2.81121608709614e-10,
4.87228709396082e-11, 8.42942705148545e-12, 1.45516736670963e-12]

# a=128 cells active out of n=4000 cells, theta=s/2
errorsA128 = [0.0629837459364841, 0.00584430512527604, 0.000596596501225661,
6.34149149902892e-5, 6.87897949491240e-6, 7.54205223721698e-7,
8.31279723703759e-8, 9.18014626543976e-9, 1.01352464978094e-9,
1.11692707533825e-10, 1.22722470782534e-11, 1.34323936472554e-12,
1.46357551285682e-13, 1.58660559533635e-14, 1.71047348036875e-15]

# a=256 cells active out of n=8000 cells, theta=s/2
errorsA256 = [0.0629798724840605, 0.00586466274476339, 0.000602973198348946,
6.47893770382337e-5, 7.13081732853977e-6, 7.96226067701095e-7,
8.97171982962439e-8, 1.01679133638258e-8, 1.15655095900038e-9,
1.31832019128216e-10, 1.50427273255654e-11, 1.71682860211354e-12,
1.95861312395311e-13, 2.23242197462621e-14, 2.54118657161951e-15]

# a=512 cells active out of n=16000 cells, theta=s/2
errorsA512 = [0.0629779361210076, 0.00587483456448912, 0.000606169580170940,
6.54824106515384e-5, 7.25889757236790e-6, 8.17841097774893e-7,
9.31547832439269e-8, 1.06920037239144e-8, 1.23394545221835e-9,
1.42977409702263e-10, 1.66151854939354e-11, 1.93490570019431e-12,
2.25664581841877e-13, 2.63454758717112e-14, 3.07765573247024e-15]

# a=32 cells active out of n=2000 cells, theta=s/2
e1 = [0.0317518759379690, 0.00145910198652343, 7.21043526586462e-5,
3.62000130960674e-6, 1.80708314241898e-7, 8.87162196669982e-9,
4.25450744299101e-10, 1.98354253495098e-11, 8.95659781541374e-13,
3.90434126921476e-14, 1.63818981220476e-15, 6.59697060273271e-17,
2.54235137209141e-18, 9.34848754480508e-20, 3.26957224042626e-21]

e1_new = [500000./500000.0, 492576./950000.0, 33719./950000.0, 1746./950000.0, 102./950000.0, 3./950000.0, 0./950000.0, 0./950000.0, 0./950000.0, 0./950000.0, 0./950000.0, 0./950000.0, 0./50000.0, 0./50000.0, 0./50000.0]

e1_new_predicted = [0.99999990152361075639022198316974800245420818812577,
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
e2 = [0.0159379844961240, 0.000368380360563884, 9.15589510774173e-6,
2.31087096485802e-7, 5.79575487730169e-9, 1.42859788831653e-10,
3.43738078417794e-12, 8.03488203503050e-14, 1.81770695269123e-15,
3.96689862829144e-17, 8.32659257455965e-19, 1.67618796222282e-20,
3.22672972195271e-22, 5.92229047076324e-24, 1.03307534135054e-25]

e2_new = [49981./50000, 8320./50000, 242./50000, 6./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000, 0./50000]


e2_new_predicted = [0.99967548539491804549604445080339348274243435001951,
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

# a=256 cells active out of n=16000 cells, theta=s/2
e3 = [0.0317449840615038, 0.00149789666286458, 7.81322668823010e-5,
4.26087025535749e-6, 2.38035717510188e-7, 1.34908860116719e-8,
7.71519657301477e-10, 4.43732367140564e-11, 2.56102362062069e-12,
1.48103355887014e-13, 8.57230129616979e-15, 4.96196289826140e-16,
2.87050187411583e-17, 1.65879779753702e-18, 9.57163117212011e-20]

# a=512 cells active out of n=16000 cells, theta=s/2
e4 = [0.0629779361210076, 0.00587483456448912, 0.000606169580170940,
6.54824106515384e-5, 7.25889757236790e-6, 8.17841097774893e-7,
9.31547832439269e-8, 1.06920037239144e-8, 1.23394545221835e-9,
1.42977409702263e-10, 1.66151854939354e-11, 1.93490570019431e-12,
2.25664581841877e-13, 2.63454758717112e-14, 3.07765573247024e-15]

e4_new = [550000./550000.0,
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


e4_new_predicted = [0.999999999999992,
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
e5 = [0.437511719482468, 0.261705563155493, 0.169396504590094,
0.113761218620382, 0.0780630164280560, 0.0543344667231431, 0.0382034690567412,
0.0270660645556014, 0.0192890442507909, 0.0138117305771507, 0.00992807501953914,
0.00715948677425288, 0.00517705699047973, 0.00375230607503995,
0.00272515634577904]

e5_new = [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.998, 0.99, 0.988, 0.914, 0.812, 0.736]

e5_new_predicted = [1.00000000000000,
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

trace1_new = Scatter(
    y=e1_new,
    x=synapses,
    line=Line(
        color='rgb(255, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=32 n=2000"
)

trace1_new_predicted = Scatter(
    y=e1_new_predicted,
    x=synapses,
    line=Line(
        color='rgb(0, 255, 0)',
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

trace2_new = Scatter(
    y=e2_new,
    x=synapses,
    line=Line(
        color='rgb(255, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=32 n=4000"
)

trace2_new_predicted = Scatter(
    y=e2_new_predicted,
    x=synapses,
    line=Line(
        color='rgb(0, 255, 0)',
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

trace4_new = Scatter(
    y=e4_new,
    x=synapses,
    line=Line(
        color='rgb(255, 0, 0)',
        width=3,
        shape='spline'
    ),
    name="a=512 n=16000"
)

trace4_new_predicted = Scatter(
    y=e4_new_predicted,
    x=synapses,
    line=Line(
        color='rgb(0, 255, 0)',
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

trace5_new = Scatter(
    y=e5_new,
    x=synapses,
    line=Line(
        color='rgb(255, 0, 0)',
        width=3,
        dash='dash',
        shape='spline',
    ),
    name="a=4000 n=16000"
)

trace5_new_predicted = Scatter(
    y=e5_new_predicted,
    x=synapses,
    line=Line(
        color='rgb(0, 255, 0)',
        width=3,
        dash='dash',
        shape='spline',
    ),
    name="a=4000 n=16000"
)

data = Data([trace1, trace1_new, trace1_new_predicted, trace2, trace2_new, trace2_new_predicted, trace4, trace4_new, trace4_new_predicted, trace5, trace5_new, trace5_new_predicted])

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
            y=0.28,
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
            y=0.47,
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
            y=0.90,
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
