#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

"""
Analyze data with wavelet transformation
"""
import numpy as np
from scipy import signal
import mne
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')
plt.ion()

def loadData(dataSet):

  if dataSet == 'sinewave':
    t = np.linspace(0, 10000*60, 2000, endpoint=False)
    T = float(14400)
    sig = np.cos(2 * np.pi * t / T) + np.random.normal(size=t.shape)*0.1
    dt = np.mean(np.diff(t))
    widths = np.logspace(0, np.log10(np.float(len(sig))/20), 50)
  elif dataSet == 'nyc_taxi':
    filePath = 'data/nyc_taxi_1min.csv'
    seq = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data', 'timeofday', 'dayofweek'])
    sig = seq['data']
    sig = sig[:20000]
    dt = 60
    t = np.array(range(len(sig)))*dt
    widths = np.logspace(0, np.log10(np.float(len(sig))/20), 50)
    # widths = np.logspace(0, np.log10(3600*24*30/dt/4), 50)
  elif dataSet == 'hotgym':
    filePath = 'data/hotgym.csv'
    seq = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['gym', 'address', 'time', 'data'])
    sig = seq['data']
    sig = sig[:10000]
    dt = 60 * 15
    t = np.array(range(len(sig)))*dt
    widths = np.logspace(0, np.log10(len(sig)/20), 50)
  elif dataSet == 'art_load_balancer':
    filePath = 'data/art_load_balancer_spikes.csv'
    seq = pd.read_csv(filePath, header=0, names=['time', 'data'])
    sig = seq['data']
    dt = 60 * 5
    t = np.array(range(len(sig)))*dt
    widths = np.logspace(0, np.log10(len(sig)/20), 50)
  elif dataSet == 'ec2_cpu_utilization_53ea38':
    filePath = 'data/ec2_cpu_utilization_53ea38.csv'
    seq = pd.read_csv(filePath, header=0, names=['time', 'data'])
    sig = seq['data']
    dt = 60 * 5
    t = np.array(range(len(sig)))*dt
    widths = np.logspace(0, np.log10(len(sig)/20), 50)
  elif dataSet == 'ec2_cpu_utilization_fe7f93':
    filePath = 'data/ec2_cpu_utilization_fe7f93.csv'
    seq = pd.read_csv(filePath, header=0, names=['time', 'data'])
    sig = seq['data']
    dt = 60 * 5
    t = np.array(range(len(sig)))*dt
    widths = np.logspace(0, np.log10(len(sig)/20), 50)
  else:
    raise NotImplementedError
  return (sig, t, dt, widths)


def plotWaveletPower(cwtmatr, time_scale, x_range=None, title=''):
  if x_range is None:
    x_range = range(0, cwtmatr.shape[1])

  fig, ax = plt.subplots(nrows=2, ncols=1)

  y_time_scale_tick = ['1-sec', '1mins', '5mins', '30mins', '60mins', '2hrs', '4hrs', '12hrs', '1day', '1week']
  y_time_scale = [1, 60, 300, 1800, 3600, 7200, 14400, 43200, 86400, 604800]

  y_tick = (np.log10(y_time_scale) - np.log10(time_scale[0]) ) / \
           (np.log10(time_scale[-1]) - np.log10(time_scale[0])) * (len(time_scale)-1)
  good_tick = np.where(np.logical_and(y_tick >= 0, y_tick < len(time_scale)))[0]
  y_tick = y_tick[good_tick]
  y_time_scale_tick = [y_time_scale_tick[i] for i in good_tick]

  ax[0].imshow(np.abs(cwtmatr[:, x_range]), aspect='auto')
  ax[0].set_yticks(y_tick)
  ax[0].set_yticklabels(y_time_scale_tick)
  ax[0].set_xlabel(' Time ')
  ax[0].set_title(title)

  ax[1].plot(sig[x_range])
  ax[1].set_xlabel(' Time ')
  ax[1].autoscale(tight=True)
  plt.show()

expDir = './result/aggregation_window_exp/'
dataset = 'ec2_cpu_utilization_fe7f93'
(sig, t, dt, widths) = loadData(dataset)
T = int(widths[-1])

# continulus wavelet transformation
cwtmatr = signal.cwt(sig, signal.ricker, widths)

cwtmatr = cwtmatr[:, 4*T:-4*T]
sig = sig[4*T:-4*T]
t = t[4*T:-4*T]
# plot wavelet coefficients along with the raw signal
freq = 1/widths.astype('float') / dt / 4
time_scale = widths * dt * 4
x_range = range(500, 1000)
plotWaveletPower(cwtmatr, time_scale, title=dataset)
plt.savefig(expDir + dataset + 'wavelet_transform.pdf')


# wavelet power spectrum
cwt_power_mean = np.mean(np.abs(cwtmatr), axis=1)
cwt_power_mean = cwt_power_mean/np.sum(cwt_power_mean)

# variance of wavelet power
cwt_power_var = np.var(np.abs(cwtmatr), axis=1)
cwt_power_var = cwt_power_var/np.sum(cwt_power_var)

cum_power_var = np.cumsum(cwt_power_var)

cwt_power_var_to_mean = cwt_power_var/cwt_power_mean


thresh = 0.01
cutoff_time_scale = time_scale[np.where(cum_power_var > thresh)[0][0]]
aggregation_time_scale = cutoff_time_scale/10
if aggregation_time_scale < dt:
  aggregation_time_scale = dt


if aggregation_time_scale <=60:
  fig_title = "suggested aggregation time scale " + '{:.6f}'.format(aggregation_time_scale) + " secs "
elif aggregation_time_scale >60:
  fig_title = "suggested aggregation time scale " + '{:.2f}'.format(aggregation_time_scale/60) + " mins "
elif aggregation_time_scale >3600:
  fig_title = "suggested aggregation time scale " + '{:.2f}'.format(aggregation_time_scale/3600) + " hours "


fig, axs = plt.subplots(nrows=2, ncols=1)
ax = axs[0]
ax.plot(time_scale, cwt_power_var, '-o')
ax.axvline(x=cutoff_time_scale, color='r')
# ax.plot(freq, cwt_power_mean)
ax.set_xscale('log')
ax.set_xlabel(' Time Scale (sec) ')
ax.set_ylabel(' Variance of Power')
ax.autoscale(tight='True')
ax.set_title(fig_title)

ax = axs[1]
ax.plot(time_scale, cum_power_var, '-o')
ax.axvline(x=cutoff_time_scale, color='r')
ax.set_xscale('log')
ax.set_xlabel(' Time Scale (sec) ')
ax.set_ylabel(' Accumulated Variance of Power')
ax.autoscale(tight='True')
plt.savefig(expDir + dataset + 'aggregation_time_scale.pdf')


