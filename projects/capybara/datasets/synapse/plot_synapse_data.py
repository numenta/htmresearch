#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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
import os
import pandas as pd

from plotly.offline import plot
from plotly.graph_objs import Scatter, Layout, XAxis, YAxis



def plot_df(df, title):
  ax = ['x', 'y', 'z']
  data = [Scatter(y=df[a], name=a) for a in ax]

  layout = Layout(title=title,
                  xaxis=XAxis(title='Timestep'),
                  yaxis=YAxis(title='{x, y, z} coordinates in Gs'))

  plot({'data': data, 'layout': layout}, show_link=False)



def main():
  data_dir = 'data'
  csv_files = ['controls_accel_walking_outbound.csv',
               'controls_accel_walking_return.csv',
               'controls_accel_walking_rest.csv',
               'has_pd_accel_walking_outbound.csv',
               'has_pd_accel_walking_return.csv',
               'has_pd_accel_walking_rest.csv', ]

  for csv_file_path in csv_files:
    csv_file = os.path.join(data_dir, csv_file_path)
    df = pd.read_csv(csv_file)
    df.describe()
    plot_df(df, csv_file_path)



if __name__ == "__main__":
  main()
