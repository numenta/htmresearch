# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

"""
Test that uses a trained deepnet encoder to create SDRs for
temporal anomaly detection.

Get Caffe here: http://caffe.berkeleyvision.org
Get data here: http://www.svcl.ucsd.edu/projects/anomaly/dataset.html
"""

from cStringIO import StringIO
import heapq
import numpy as np
import scipy.ndimage as nd
import PIL.Image
import os
import caffe

from IPython.display import clear_output, Image, display
from google.protobuf import text_format
from matplotlib import pyplot as plt


from nupic.algorithms import anomaly
from nupic.research.temporal_memory import TemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
    TemporalMemoryMonitorMixin)


class MonitoredTemporalMemory(TemporalMemoryMonitorMixin,
							  TemporalMemory):
	pass

training_trials = 20

caffe_root = '/Users/tsilver/caffe/' # substitute your path here
model_path = caffe_root+'models/bvlc_googlenet/'
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]),
                       channel_swap = (2,1,0))


tm = MonitoredTemporalMemory(columnDimensions=(1024,))

def encodeImg(img_file, net, d=250, end = 'pool5/7x7_s1'):

	src = net.blobs['data']
	img = PIL.Image.open(img_file)
	img = np.float32(img.resize((d, d), PIL.Image.ANTIALIAS))
	src.reshape(1,3,d,d)
	src.data[0] = img

	net.forward(end=end)
	last_hidden_layer = net.blobs[end].data[0,:,0,0]

	n = len(last_hidden_layer)
	w = int(0.02*n)
	theta = heapq.nlargest(w, last_hidden_layer)[-1]

	return set(np.where(np.asarray(last_hidden_layer) >= theta)[0])

def plotBursts(tm, title=None):
	ys = [len(x) for x in tm.mmGetTraceUnpredictedActiveColumns().data]
	fig, ax = plt.subplots()

	index = np.arange(len(ys))

	bar_width = 1.0

	opacity = 0.4

	rects1 = plt.bar(index, ys, bar_width,
	               alpha=opacity,
	               color='b',
	               label='No Feedback')

	plt.xlabel('Time')
	plt.ylabel('# Bursting Columns')
	if title:
		plt.title(title)

	plt.tight_layout()
	plt.show()

def plotAnomalyScores(tm, title=None):
	totalActiveColumns = [len(x) for x in tm.mmGetTraceActiveColumns().data]
	unpredictedColumns = [len(x) for x in tm.mmGetTraceUnpredictedActiveColumns().data]
	anomalyScores = []

	xs = range(len(unpredictedColumns))

	for i in xs:
		anomalyScores.append((1.0*unpredictedColumns[i])/totalActiveColumns[i])

	print anomalyScores

	plt.plot(xs, anomalyScores)

	plt.xlabel('Time')
	plt.ylabel('Raw Anomaly Score')

	plt.tight_layout()
	plt.show()

def computeDirectory(dirPath, net, tm, start=1, end=201, learn=True):
	for i in range(start, end):
		img_name = str(i).zfill(3)+'.tif'
		img_path = os.path.join(dirPath, img_name)
		print "Processing ",
		print img_path
		sdr = encodeImg(img_path, net)
		tm.compute(sdr, learn=learn)

trainingDirPaths = []
for i in range(1, 35):
	trainingDirPaths.append('UCSDped1/Train/Train'+str(i).zfill(3))

testingDirPaths = []
for i in range(1, 37):
	testingDirPaths.append('UCSDped1/Test/Test'+str(i).zfill(3))

for i in range(training_trials):
	print "Starting Trial ", str(i)
	for dirPath in trainingDirPaths:
		print "Processing directory ",
		print dirPath
		computeDirectory(dirPath, net, tm)
		print
	print

	tm.mmClearHistory()

print "Starting testing"
for dirPath in testingDirPaths:
	print "Processing directory ",
	print dirPath
	computeDirectory(dirPath, net, tm, learn=False)
	print

plotAnomalyScores(tm)
