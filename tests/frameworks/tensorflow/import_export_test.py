# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
Test import/export models between pytorch and tensorflow
"""
import os
import random
import shutil
import tempfile

import numpy as np
import onnx
import onnx_tf
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import keras
from torchvision import datasets, transforms

BATCH_SIZE = 64
EPOCHS = 5
SEED = 42
N = 100
LEARNING_RATE = 0.02
MOMENTUM = 0.25

# Tensorflow configuration.
# Make sure to use one thread in order to keep the results deterministic
CONFIG = tf.ConfigProto(
  intra_op_parallelism_threads=1,
  inter_op_parallelism_threads=1,
  device_count={'CPU': 1}
)



class PytorchMNIST(nn.Module):
  """
  Simple pytorch MNIST Model
  """


  def __init__(self, n):
    super(PytorchMNIST, self).__init__()
    self.n = n
    self.l1 = nn.Linear(28 * 28, self.n)
    self.l2 = nn.Linear(self.n, 10)


  def forward(self, x):
    x = self.l1(x)
    x = F.relu(x)
    x = self.l2(x)
    x = F.log_softmax(x, dim=1)
    return x



class ImportExportModelTest(tf.test.TestCase):

  def setUp(self):
    self.tmpDir = tempfile.mkdtemp()
    self.dataDir = os.path.join(self.tmpDir, "data")
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


  def tearDown(self):
    shutil.rmtree(self.tmpDir)


  def _train_pytorch(self, model):
    """
    Train pytorch model using MNIST Dataset
    :param model: PytorchMNIST model
    """
    data_loader = torch.utils.data.DataLoader(
      datasets.MNIST(self.dataDir, train=True, download=True,
                     transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
      batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LEARNING_RATE,
                                momentum=MOMENTUM)
    model.train()
    for epoch in xrange(EPOCHS):
      for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.view(-1, 28 * 28)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


  def _test_pytorch(self, model):
    """
    Test pre-trained pytorch model using MNIST Dataset
    :param model: Pre-trained PytorchMNIST model
    :return: tuple(loss, accuracy)
    """
    data_loader = torch.utils.data.DataLoader(
      datasets.MNIST(self.dataDir, train=False, download=True,
                     transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
      batch_size=BATCH_SIZE, shuffle=True)

    model.eval()
    loss = 0.0
    num_correct = 0.0
    with torch.no_grad():
      for data, target in data_loader:
        data = data.view(-1, 28 * 28)
        output = model(data)
        loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        num_correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    accuracy = num_correct / len(data_loader.dataset)

    return (loss, accuracy)


  def testImportFromPytorch(self):
    """
    Test importing a pre-trained pytorch model into tensorflow for inference
    """

    # Export trained pytorch model to onnx
    pt_model = PytorchMNIST(N)
    self._train_pytorch(pt_model)
    dummy_input = torch.autograd.Variable(torch.FloatTensor(1, 28 * 28))
    torch.onnx.export(pt_model, dummy_input,
                      os.path.join(self.tmpDir, "torch.onnx"),
                      input_names=["input"] + pt_model.state_dict().keys(),
                      output_names=["output"])

    _, expected = self._test_pytorch(pt_model)

    # Load ONNX model
    model = onnx.load(os.path.join(self.tmpDir, "torch.onnx"))
    # HACK: Convert model's input shape to dynamic, ignoring batch size
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    onnx.checker.check_model(model)

    tf_model = onnx_tf.backend.prepare(model)
    # Load MNIST test data
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28 * 28) / 255.0
    x_test = (x_test - 0.1307) / 0.3081

    with self.test_session(config=CONFIG):
      # Compute accuracy
      output = tf_model.run(x_test)
      predictions = output[0].argmax(axis=1)
      accuracy = tf.reduce_mean(tf.cast(predictions == y_test, tf.float32))

      self.assertAlmostEqual(accuracy.eval(), expected, places=4)



if __name__ == "__main__":
  tf.test.main()
