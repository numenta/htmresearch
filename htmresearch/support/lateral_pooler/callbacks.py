# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
import numpy as np
import pickle
import sys


class Callback(object):
  """
  An abstract callback class. 
  The "model" property of the class will be set
  with in the "fit" loop of the pooler model.
  """
  def set_model(self, model):
      self.model = model

  def on_epoch_begin(self, epoch, cache):
      pass

  def on_epoch_end(self, epoch, cache):
      pass

  def on_batch_begin(self, batch, cache):
      pass

  def on_batch_end(self, batch, cache):
      pass


class ModelCheckpoint(Callback):
  """
  Callback the saves the model after every epoch.
  """
  def __init__(self, filename):
      self.filename = filename

  def on_epoch_end(self, epoch, cache):
      model = self.model
      filename = self.filename.format(epoch + 1)
      with open(filename, 'wb') as f:
          pickle.dump(model, f)


class ModelOutputEvaluator(Callback):
  """
  Callback that evaluates the model after every epoch.
  """
  def __init__(self, metric, data, on_batch = False ):
      self.on_batch = on_batch
      self.model  = None
      self.test_data   = data
      self.metric = metric
      self.results = []

  def evaluate_output(self, cache):
      model  = self.model
      metric = self.metric
      X      = self.test_data
      Y      = model.encode(X)
      self.results.append(metric(Y))

  def on_batch_end(self, batch, cache):
      if self.on_batch:
          self.evaluate_output(cache)

  def on_epoch_end(self, epoch, cache):
      self.evaluate_output(cache)

  def get_results(self):
      return np.array(self.results)


class OutputCollector(Callback):
  """
  Callback that evaluates the model after every epoch.
  """
  def __init__(self):
      self.inputs   = []
      self.outputs  = []

  def on_epoch_begin(self, epoch, cache):
      self.inputs.append([])
      self.outputs.append([])

  def on_batch_end(self, batch, cache):
      X, Y = batch
      self.inputs[-1].append(X)
      self.outputs[-1].append(Y)

  def on_epoch_end(self, epoch, cache):
      self.outputs[-1] = np.bmat(self.outputs[-1])
      self.inputs[-1]  = np.bmat(self.inputs[-1])

  def get_inputs(self):
      return np.array(self.inputs)

  def get_outputs(self):
      return np.array(self.outputs)



class ModelInspector(Callback):
  """
  Callback that evaluates the model after every epoch.
  """
  def __init__(self, inspect, on_batch = False ):
      self.on_batch = on_batch
      self.model    = None
      self.inspect  = inspect
      self.results  = []

  def inspect_model(self, cache):
      model   = self.model
      inspect = self.inspect
      self.results.append(inspect(model))

  def on_batch_end(self, batch, cache):
      if self.on_batch == True:
          self.inspect_model(cache)

  def on_epoch_end(self, epoch, cache):
      if self.on_batch == False:
          self.inspect_model(cache)

  def get_results(self):
      return np.array(self.results)


class Reconstructor(Callback):
  """
  Callback that evaluates the model after every epoch.
  """
  def __init__(self, data, on_batch = False ):
      self.model  = None
      self.test_data   = data
      self.results = []
      self.outputs = []
      self.on_batch = on_batch

  def reconstruct(self, cache):
      model = self.model

      # W     = model.visible_to_hidden

      W = model.feedforward
      X     = self.test_data
      Y     = model.encode(X)
      m, d  = X.shape
      X_rec = np.dot(W.T,Y)

      self.outputs.append(Y)
      self.results.append(X_rec)

  def on_epoch_end(self, epoch, cache):
      self.reconstruct(cache)

  def on_batch_end(self, batch, cache):
      if self.on_batch:
          self.reconstruct(cache)

  def get_results(self):
      return np.array(self.results)

  def get_outputs(self):
      return np.array(self.outputs)



class Logger(Callback):
  """
  Prints at what epoch and minibatch we are in training.
  """
  def __init__(self):
      self.model  = None

  def on_epoch_begin(self, epoch, cache):
    cache["epoch"] = epoch
    cache["batch"] = 0

  def on_batch_begin(self, batch, cache):
    sys.stdout.flush()

    epoch = cache["epoch"]
    t = cache["batch"] + 1
    cache["batch"] += 1
    num_epochs = cache["num_epochs"]
    d = cache["num_batches"]


    sys.stdout.write(
      "\r{}/{}  {}/{}"
        .format(num_epochs, epoch + 1, d, t + 1))


