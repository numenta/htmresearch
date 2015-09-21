# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
"""
This file contains plotting tools for NLP experiment results.
"""

import math
import numpy
import os
import pandas as pd
import plotly.plotly as py
import plotly.tools as tls

from plotly.graph_objs import (
    Data,
    ErrorY,
    Figure,
    Font,
    Heatmap,
    Layout,
    Margin,
    Scatter,
    XAxis,
    YAxis)



class PlotNLP():
  """Class to plot evaluation metrics for NLP experiments."""

  def __init__(self,
               apiKey=None,
               username=None,
               experimentName="experiment"):
    # Instantiate API credentials.
    try:
      self.apiKey = apiKey if apiKey else os.environ["PLOTLY_API_KEY"]
    except:
      print ("Missing PLOTLY_API_KEY environment variable. If you have a "
        "key, set it with $ export PLOTLY_API_KEY=api_key\n"
        "You can retrieve a key by registering for the Plotly API at "
        "http://www.plot.ly")
      raise OSError("Missing API key.")
    try:
      self.username = username if username else os.environ["PLOTLY_USERNAME"]
    except:
      print ("Missing PLOTLY_USERNAME environment variable. If you have a "
        "username, set it with $ export PLOTLY_USERNAME=username\n"
        "You can sign up for the Plotly API at http://www.plot.ly")
      raise OSError("Missing username.")

    py.sign_in(self.username, self.apiKey)

    self.experimentName = experimentName


  @staticmethod
  def getDataFrame(dataPath):
    """Get pandas dataframe of the results CSV."""
    try:
      return pd.read_csv(dataPath)
    except IOError("Invalid data path to file"):
      return


  @staticmethod
  def interpretConfusionMatrixData(dataFrame, normalize):
    """Parse pandas dataframe into confusion matrix format."""
    labels = dataFrame.columns.values.tolist()[:-1]
    values = map(list, dataFrame.values)

    for i, row in enumerate(values):
      values[i] = [v/row[-1] for v in row[:-1]] if normalize else row[:-1]
    cm = {"x":labels,
          "y":labels[:-1],
          "z":values[:-1]
          }
    return cm


  def plotConfusionMatrix(self, data, normalize=True):
    """
    Plots the confusion matrix of the input dataframe.
    
    @param data         (pandas DF)     The confusion matrix.

    @param normalize    (bool)          True will normalize the confusion matrix
        values for the total number of actual classifications per label. Thus
        the cm values are 0.0 to 1.0.
    """
    xyzData = self.interpretConfusionMatrixData(data, normalize)

    data = Data([Heatmap(z=xyzData["z"],
                         x=xyzData["x"],
                         y=xyzData["y"],
                         colorscale='YIGnBu')])

    layout = Layout(
      title='Confusion matrix for ' + self.experimentName,
      xaxis=XAxis(
        title='Predicted label',
        side='top',
        titlefont=Font(
          family='Courier New, monospace',
          size=18,
          color='#7f7f7f'
        )
      ),
      yaxis=YAxis(
        title='True label',
        titlefont=Font(
          family='Courier New, monospace',
          size=18,
          color='#7f7f7f'
        ),
        autorange='reversed'
      ),
      barmode='overlay',
      autosize=True,
      width=1000,
      height=1000,
      margin=Margin(
        l=200,
        r=80,
        b=80,
        t=450
        )
    )

    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig)
    print "Confusion matrix URL: ", plot_url


  def plotCategoryAccuracies(self, trialAccuracies, trainSizes):
    """
    Shows the accuracy for the categories at a certain training size
    
    @param trialAccuracies    (dict)    A dictionary of dictionaries. For each
        train size, there is a dictionary that maps a category to a list of 
        accuracies for that category.
    
    @param trainSizes         (list)    Size of training set for each trial.
    """
    sizes = sorted(set(trainSizes))
    size_sqrt = math.sqrt(len(sizes))
    subplotDimension = int(math.ceil(size_sqrt))

    rows = subplotDimension
    cols = subplotDimension
    if len(sizes) <= subplotDimension * (subplotDimension - 1):
      rows -= 1

    fig = tls.make_subplots(rows=rows, cols=cols,
      shared_xaxes=True, shared_yaxes=True, print_grid=False)
    num_categories = 0
    for i, s in enumerate(sizes):
      # 1-indexed
      col = i % cols + 1
      row = (i - col + 1) / cols + 1
      classificationAccuracies = trialAccuracies[s]
      num_categories = max(num_categories,len(classificationAccuracies.keys()))
      
      x = []
      y = []
      std = []
      for label, acc in classificationAccuracies.iteritems():
        x.append(label)
        y.append(numpy.mean(acc))
        std.append(numpy.std(acc))

      trace = Scatter(
        x=x,
        y=y,
        name=s,
        mode='markers',
        error_y=ErrorY(
          type='data',
          symmetric=False,
          array=std,
          arrayminus=std,
          visible=True
        )
      )

      fig.append_trace(trace, row, col)

    fig["layout"]["title"] = "Accuracies for category by training size"
    half_way_cols =  int(math.ceil(cols / 2.0))
    half_way_rows =  int(math.ceil(rows / 2.0))
    fig["layout"]["xaxis{}".format(half_way_cols)]["title"] = "Category Label"
    fig["layout"]["yaxis{}".format(half_way_rows)]["title"] = "Accuracy"
    for i in xrange(1, cols + 1):
      fig["layout"]["xaxis{}".format(i)]["tickangle"] = -45
      fig["layout"]["xaxis{}".format(i)]["nticks"] = num_categories * 2
      if i <= rows:
        fig["layout"]["yaxis{}".format(i)]["range"] = [-.1, 1.1]
    fig["layout"]["margin"] = {"b" : 120}

    plot_url = py.plot(fig)
    print "Category Accuracies URL: ", plot_url


  def plotCumulativeAccuracies(self, classificationAccuracies, trainSizes):
    """
    Creates scatter plots that show the accuracy for each category at a
    certain training size
    
    @param classificationAccuracies (dict)    Maps a category label to a list of
        lists of accuracies. Each item in the key is a list of accuracies for
        a specific training size, ordered by increasing training size.
        
    @param trainSizes                (list)    Sizes of training sets for trials.
    """
    # Convert list of list of accuracies to list of means
    classificationSummaries = [(label, map(numpy.mean, acc))
        for label, acc in classificationAccuracies.iteritems()]
    
    data = []
    sizes = sorted(set(trainSizes))
    for label, summary in classificationSummaries:
      data.append(Scatter(x=sizes, y=summary, name=label))
    data = Data(data)

    layout = Layout(
      title='Cumulative Accuracies for ' + self.experimentName,
      xaxis=XAxis(
        title='Training size',
        titlefont=Font(
          family='Courier New, monospace',
          size=18,
          color='#7f7f7f'
        )
      ),
      yaxis=YAxis(
        title='Accuracy',
        titlefont=Font(
          family='Courier New, monospace',
          size=18,
          color='#7f7f7f'
        )
      )
    )

    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig)
    print "Cumulative Accuracies URL: ", plot_url
