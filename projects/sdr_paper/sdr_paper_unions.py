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

from sympy import *
init_printing()
from IPython.display import display

import plotly.plotly as py
from plotly.graph_objs import *
import os


# Define the (global) equations
# Equation from Neuron paper
#
#       A dendritic segment can robustly classify a pattern by
#       subsampling a small number of cells from a larger population.  Assuming
#       a random distribution of patterns, the exact probability of a false
#       match is given by the following equation: where n refers to the size of
#       the population of cells, a is the number of active cells at any instance
#       in time, s is the number of actual synapses on a dendritic segment, and
#       theta is the threshold for NMDA spikes. Following  (Ahmad & Hawkins,
#       2015), the numerator counts the number of possible ways theta or more
#       cells can match a fixed set of s synapses. The denominator counts the
#       number of ways a cells out of n can be active.
#

b, n, s, a, theta = symbols("b n s a theta")

subsampledOmega = (binomial(s, b) * binomial(n - s, a - b)) / binomial(n, a)
subsampledFpF = Sum(subsampledOmega, (b, theta, s))
subsampledOmegaSlow = (binomial(s, b) * binomial(n - s, a - b))
subsampledFpFSlow = Sum(subsampledOmegaSlow, (b, theta, s))/ binomial(n, a)

display(subsampledFpF)
display(subsampledFpFSlow)

# Union formula
#
# Formula for calculating the number of bits that are ON after M union
# operations. This number can then be used in the above equations to calculate
# error probabilities with unions.

# Probability a given bit is 0 after M union operation
M = Symbol("M")
p0 = Pow((1-(s/n)),M)

# Expected number of ON bits after M union operations.
numOnBits = (1-p0)*n


# If a dendrite has s synapses, and the activity at time t is a union of
# M patterns, plot probability of a false match as a function of M
#
def falseMatchDendritevsMPatterns(n_=1000):
  a_ = 30
  theta_ = 10
  s_ = 20

  # Arrays used for plotting
  MList = []
  errorList = []

  print "\n\nn=%d, a=%d, theta=%d, s=%d" % (n_,a_,theta_,s_)

  error = 0.0
  for M_ in range(1, 40):

    # Number of bits active in At
    atAfterUnion = numOnBits.subs(n, n_).subs(s, a_).subs(M,M_).evalf()
    if error >= 0.9999999999:
      error = 1.0
    else:
      if M_ <= 8:
        eq3 = subsampledFpFSlow.subs(n, n_).subs(a, atAfterUnion).subs(theta, theta_)
      else:
        eq3 = subsampledFpF.subs(n, n_).subs(a, atAfterUnion).subs(theta, theta_)
      error = eq3.subs(s,s_).evalf()

    print M_,atAfterUnion,error

    MList.append(M_)
    errorList.append(error)

  print MList
  print errorList
  return MList,errorList


# Plot probability of a false match as a function of M
def falseMatchvsM(n_=1000):
  a_ = 200
  theta_ = 15
  s_ = 25

  # Arrays used for plotting
  MList = []
  errorList = []

  print "\n\nn=%d, a=%d, theta=%d, s=%d" % (n_,a_,theta_,s_)

  error = 0.0
  for M_ in range(1, 20):

    # Need this otherwise calculation goes to 0
    numSynapsesAfterUnion = numOnBits.subs(n, n_).subs(s, s_).subs(M,
                                                                   M_).evalf()
    if error >= 0.9999999999:
      error = 1.0
    else:
      if M_ <= 2:
        eq3 = subsampledFpFSlow.subs(n, n_).subs(a, a_).subs(theta, theta_)
      else:
        eq3 = subsampledFpF.subs(n, n_).subs(a, a_).subs(theta, theta_)
      error = eq3.subs(s,round(numSynapsesAfterUnion)).evalf()

    print M_,numSynapsesAfterUnion,error

    MList.append(M_)
    errorList.append(error)

  print MList
  print errorList
  return MList,errorList


def plotFalseMatchvsM():
  # Plot probability of a false match as a function of M
  # Plot three different values of n. n=1000, 10000, 20000?

  # listofMValues, errorsN1000 = falseMatchDendritevsMPatterns(1000)
  # listofMValues, errorsN4000 = falseMatchDendritevsMPatterns(4000)
  # listofMValues, errorsN8000 = falseMatchDendritevsMPatterns(8000)

  # I get a JSON serialization error if I try to use the above values directly
  # Go figure!
  listofMValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
  37, 38, 39]


  errorsN1000 = [1.74966917747145e-11, 2.82658811059119e-8, 1.39651026490819e-6,
  1.85276803498981e-5, 0.000122462323694764, 0.000525455686291461,
  0.00168037428756938, 0.00434432548818933, 0.00956421362220440,
  0.0185736190367658, 0.0326241695265783, 0.0527946135484057,
  0.0798227334712682, 0.113993339287459, 0.155097506782092, 0.202460851761100,
  0.255026162466195, 0.311469519214058, 0.370328446983243, 0.430123973179965,
  0.489463792982226, 0.547119475740302, 0.602075692288738, 0.653553242557473,
  0.701010071550924, 0.744125611399872, 0.782773952468071, 0.816990843307732,
  0.846938639252940, 0.872872296277516, 0.895108505724166, 0.913999187918402,
  0.929909857645548, 0.943202853231983, 0.954225070369372, 0.963299635403854,
  0.970720859072523, 0.976751799514041, 0.981623805319142]

  errorsN4000 = [1.85597138333997e-17, 4.17574792548866e-14,
  2.84761986763360e-12, 5.19578251035665e-11, 4.70969827213147e-10,
  2.76394389581973e-9, 1.20571177807438e-8, 4.24050541935861e-8,
  1.26640697074757e-7, 3.32635948387612e-7, 7.87813300982585e-7,
  1.71350339989923e-6, 3.47030215618841e-6, 6.61498904948652e-6,
  1.19688817202398e-5, 2.06967802112776e-5, 3.43949533871793e-5,
  5.51859885899797e-5, 8.58178016036501e-5, 0.000129763714131879,
  0.000191320265399264, 0.000275699337707405, 0.000389111238164987,
  0.000538835578257418, 0.000733277111714841, 0.000982004107773168,
  0.00129576732778743, 0.00168649821399632, 0.00216728546638854,
  0.00275232975498735, 0.00345687687063695, 0.00429713014050859,
  0.00529014341106307, 0.00645369632028853, 0.00780615393496547,
  0.00936631311273528, 0.0111532381608081, 0.0131860885035822,
  0.0154839411426565]

  errorsN8000 = [1.84408018909783e-20, 4.38205425485483e-17,
  3.15228206823515e-15, 6.06524481280268e-14, 5.79646066756809e-13,
  3.58598795197947e-12, 1.64883751834296e-11, 6.11160288580599e-11,
  1.92337498577987e-10, 5.32308966989701e-10, 1.32822871042413e-9,
  3.04326996437801e-9, 6.49201569794134e-9, 1.30331279847690e-8,
  2.48330734422034e-8, 4.52154349885235e-8, 7.91106310476925e-8,
  1.33620715829301e-7, 2.18713348143796e-7, 3.48058004061341e-7,
  5.40016098895161e-7, 8.18794913513682e-7, 1.21577313632138e-6,
  1.77100348692262e-6, 2.53489533726787e-6, 3.57007754920002e-6,
  4.95343896161834e-6, 6.77834114373759e-6, 9.15699523688480e-6,
  1.22229919872450e-5, 1.61339714722548e-5, 2.10744165854308e-5,
  2.72585521047319e-5, 3.49333291580426e-5, 4.43814731417444e-5,
  5.59245716630590e-5, 6.99261778781310e-5, 8.67949036939708e-5,
  0.000106987476696977]

  # listofMValues,errorsN1000 = falseMatchvsM(1000)
  # listofMValues,errorsN10000 = falseMatchvsM(10000)
  # listofMValues,errorsN20000 = falseMatchvsM(20000)
  #
  # # I get a JSON serialization error if I try to use the above values directly
  # # Go figure!
  #
  # listofMValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  #                  18, 19]
  #
  # errorsN1000 = [1.00140315240769e-5, 0.0473894530292855, 0.501650232353378,
  # 0.899154755459308, 0.991079536716752, 0.999523524650922, 0.999981553264809,
  # 0.999999512691584, 0.999999990860972, 0.999999999845828, 0.999999999997526, 1.0,
  # 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  #
  # errorsN10000 = [5.30352419025194e-20, 2.35602184920806e-14,
  # 1.53934243906984e-11, 1.10302867119927e-9, 2.27921915526762e-8,
  # 2.69253218125502e-7, 1.98072351729197e-6, 9.75307801058064e-6,
  # 3.98373263650013e-5, 0.000127080441383853, 0.000362630253250421,
  # 0.000874610762751977, 0.00190208641223641, 0.00378812896412743,
  # 0.00716447200395456, 0.0123542384410630, 0.0201165770143612, 0.0311403472140338,
  # 0.0453789479134073]
  #
  # errorsN20000 = [1.75826328738259e-24, 9.73603540551588e-19,
  # 7.92130752699486e-16, 7.08428692442973e-14, 2.03825914561597e-12,
  # 2.93568736488724e-11, 2.44471331836285e-10, 1.59493239439383e-9,
  # 8.04780894887436e-9, 3.32242567600452e-8, 1.11328056326208e-7,
  # 3.44426535137730e-7, 9.53448550682057e-7, 2.31864104977855e-6,
  # 5.40383164541756e-6, 1.13919680675529e-5, 2.33141516738363e-5,
  # 4.40336400781654e-5, 8.14089056445047e-5]


  plotlyUser = os.environ['PLOTLY_USERNAME']
  plotlyAPIKey = os.environ['PLOTLY_API_KEY']
  py.sign_in(plotlyUser, plotlyAPIKey)


  trace1 = Scatter(
      y=errorsN1000,
      x=listofMValues,
      line=Line( color='rgb(0, 0, 0)', width=3, shape='spline' ),
      name="n=1000"
  )

  trace2 = Scatter(
      y=errorsN4000,
      x=listofMValues,
      line=Line( color='rgb(0, 0, 0)', width=3, shape='spline' ),
      name="n=4000"
  )

  trace3 = Scatter(
      y=errorsN8000,
      x=listofMValues,
      line=Line( color='rgb(0, 0, 0)', width=3, shape='spline' ),
      name="n=8000"
  )

  data = Data([trace1, trace2, trace3])

  layout = Layout(
      title='', showlegend=False, autosize=False, width=855, height=700,
      xaxis=XAxis(
          title='$\\text{Number of combined patterns } (M)$',
          titlefont=Font( family='', size=16, color='' ),
          tickfont=Font( family='', size=16, color='' ),
          exponentformat="none",
          dtick=2,
          showline=True,
          range=[0,20],
      ),
      yaxis=YAxis(
          title='Probability of false positives',
          type='log',
          exponentformat='power',
          autorange=True,
          titlefont=Font( family='', size=18, color='' ),
          tickfont=Font( family='', size=12, color='' ),
          showline=True,
      ),
      annotations=Annotations([
        Annotation(
          x=10,
          y=0.1243,
          xref='x',
          yref='paper',
          text='$s = 20, \\theta=10, a=30$',
          showarrow=False,
          font=Font(
            family='Arial',
            size=20,
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
          x=7.7906, y=0.9,
          xref='x',
          yref='paper',
          text='$n = 1000$',
          showarrow=False,
          font=Font( family='', size=16, color='' ),
          align='center',
          textangle=0,
          bordercolor='', borderwidth=1, borderpad=1,
          bgcolor='rgba(0, 0, 0, 0)',
          opacity=1
        ),
        Annotation(
          x=7.906, y=0.6548,
          xref='x',
          yref='paper',
          text='$n = 4000$',
          showarrow=False,
          font=Font( family='', size=16, color='' ),
          align='center',
          textangle=0,
          bordercolor='', borderwidth=1, borderpad=1,
          bgcolor='rgba(0, 0, 0, 0)',
          opacity=1
        ),
        Annotation(
          x=7.906, y=0.44,
          xref='x',
          yref='paper',
          text='$n = 8000$',
          showarrow=False,
          font=Font( family='', size=16, color='' ),
          align='center',
          textangle=0,
          bordercolor='', borderwidth=1, borderpad=1,
          bgcolor='rgba(0, 0, 0, 0)',
          opacity=1
        ),
      ]),
  )

  fig = Figure(data=data, layout=layout)
  plot_url = py.plot(fig, auto_open=False)
  print "url=",plot_url
  figure = py.get_figure(plot_url)
  py.image.save_as(figure, 'union_effect_of_n.pdf', scale=1)


def synapsesvsM(s_ = 25):
  # Compute number of synapses on a segment as a function of M
  listofMValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19]
  listofSynapses = []
  n_=1000
  print "\n\nn=%d, s=%d" % (n_,s_)
  for M_ in listofMValues:
    numSynapsesAfterUnion = numOnBits.subs(n,n_).subs(s,s_).subs(M,M_).evalf()
    listofSynapses.append(numSynapsesAfterUnion)

  print listofSynapses

def plotSynapsesvsM():
  # Plot number of synapses on a segment as a function of M

  listofMValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19]

  synapsesS20 = [20.0000000000000, 39.6000000000000, 58.8080000000000,
  77.6318400000000, 96.0792032000000, 114.157619136000, 131.874466753280,
  149.236977418214, 166.252237869850, 182.927193112453, 199.268649250204,
  215.283276265200, 230.977610739896, 246.358058525098, 261.430897354596,
  276.202279407504, 290.678233819354, 304.864669142967, 318.767375760108]

  synapsesS25 = [25.0000000000000, 49.3750000000000, 73.1406250000000,
  96.3121093750000, 118.904306640625, 140.931698974609, 162.408406500244,
  183.348196337738, 203.764491429295, 223.670379143562, 243.078619664973,
  262.001654173349, 280.451612819015, 298.440322498540, 315.979314436076,
  333.079831575174, 349.752835785795, 366.009014891150, 381.858789518871]

  synapsesS30 = [30.0000000000000, 59.1000000000000, 87.3270000000000,
  114.707190000000, 141.265974300000, 167.027995071000, 192.017155218870,
  216.256640562304, 239.768941345435, 262.575873105072, 284.698596911920,
  306.157639004562, 326.972909834425, 347.163722539392, 366.748810863211,
  385.746346537314, 404.173956141195, 422.048737456959, 439.387275333250]


  plotlyUser = os.environ['PLOTLY_USERNAME']
  plotlyAPIKey = os.environ['PLOTLY_API_KEY']
  py.sign_in(plotlyUser, plotlyAPIKey)


  trace1 = Scatter(
      y=synapsesS20,
      x=listofMValues,
      line=Line( color='rgb(0, 0, 0)', width=3, shape='spline' ),
      name="s=20"
  )

  trace2 = Scatter(
      y=synapsesS25,
      x=listofMValues,
      line=Line( color='rgb(0, 0, 0)', width=3, shape='spline' ),
      name="s=25"
  )

  trace3 = Scatter(
      y=synapsesS30,
      x=listofMValues,
      line=Line( color='rgb(0, 0, 0)', width=3, shape='spline' ),
      name="s=30"
  )

  data = Data([trace1, trace2, trace3])

  layout = Layout(
      title='', showlegend=False, autosize=False, width=855, height=700,
      xaxis=XAxis(
          title='$\\text{Number of combined patterns } (M)$',
          titlefont=Font( family='', size=16, color='' ),
          tickfont=Font( family='', size=16, color='' ),
          exponentformat="none",
          dtick=2,
          range=[0,20],
      ),
      yaxis=YAxis(
          title='Number of synapses on segment',
          exponentformat='none',
          autorange=True,
          titlefont=Font( family='', size=18, color='' ),
          tickfont=Font( family='', size=12, color='' ),
      ),
      annotations=Annotations([
        Annotation(
          x=18.0, y=0.616,
          xref='x',
          yref='paper',
          text='$s = 20$',
          showarrow=False,
          font=Font( family='', size=16, color='' ),
          align='center',
          textangle=0,
          bordercolor='', borderwidth=1, borderpad=1,
          bgcolor='rgba(0, 0, 0, 0)',
          opacity=1
        ),
        Annotation(
          x=18.0, y=0.7552,
          xref='x',
          yref='paper',
          text='$s = 25$',
          showarrow=False,
          font=Font( family='', size=16, color='' ),
          align='center',
          textangle=0,
          bordercolor='', borderwidth=1, borderpad=1,
          bgcolor='rgba(0, 0, 0, 0)',
          opacity=1
        ),
        Annotation(
          x=18.0, y=0.876,
          xref='x',
          yref='paper',
          text='$s = 30$',
          showarrow=False,
          font=Font( family='', size=16, color='' ),
          align='center',
          textangle=0,
          bordercolor='', borderwidth=1, borderpad=1,
          bgcolor='rgba(0, 0, 0, 0)',
          opacity=1
        ),
      ]),
  )

  fig = Figure(data=data, layout=layout)
  plot_url = py.plot(fig, auto_open=False)
  print "url=",plot_url
  figure = py.get_figure(plot_url)
  py.image.save_as(figure, 'union_number_synapses.pdf', scale=1)



plotFalseMatchvsM()
# plotSynapsesvsM()

# falseMatchDendritevsMPatterns(1000)
# falseMatchDendritevsMPatterns(4000)
# falseMatchDendritevsMPatterns(8000)
