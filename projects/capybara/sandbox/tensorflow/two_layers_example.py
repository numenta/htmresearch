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

import tensorflow as tf
import time
import shutil

learning_rate = 0.1
num_iterations = 100000
print_period = 1000
num_output_labels = 2

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

X_width = len(X[0])
y_width = len(y[0])

# Network
X_ = tf.placeholder(tf.float32, shape=[None, X_width], name='X')
y_ = tf.placeholder(tf.float32, shape=[None, y_width], name='y')

W1 = tf.Variable(tf.random_uniform([X_width, X_width]), name="W1")
W2 = tf.Variable(tf.random_uniform([X_width, y_width]), name="W2")

b1 = tf.Variable(tf.zeros([X_width]), name="b1")
b2 = tf.Variable(tf.zeros([y_width]), name="b2")

# Summaries for TensorBoard
w1_h = tf.histogram_summary("W1", W1)
b1_h = tf.histogram_summary("b1", b1)
w2_h = tf.histogram_summary("W2", W2)
b2_h = tf.histogram_summary("b2", b2)

# Scopes are also for TensorBoard
with tf.name_scope("layer1") as layer1_scope:
  H1 = tf.sigmoid(tf.matmul(X_, W1) + b1)

with tf.name_scope("layer2") as layer2_scope:
  prediction_op = tf.sigmoid(tf.matmul(H1, W2) + b2)

with tf.name_scope("loss") as loss_scope:
  loss_op = tf.reduce_mean(((y_ * tf.log(prediction_op)) +
                            ((1 - y_) * tf.log(1.0 - prediction_op))) * -1)
  tf.scalar_summary("loss", loss_op)

with tf.name_scope("train") as scope:
  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)
  # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

# Initialize network
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

summary_op = tf.merge_all_summaries()
out_dir = "/tmp/tf/logs/example"
shutil.rmtree(out_dir, ignore_errors=True)
writer = tf.train.SummaryWriter(out_dir, sess.graph_def)

t_start = time.clock()
for i in range(num_iterations):
  _, summary = sess.run([train_op, summary_op], feed_dict={X_: X, y_: y})

  writer.add_summary(summary, i)
  if i % print_period == 0:
    predictions = sess.run(prediction_op, feed_dict={X_: X, y_: y})
    loss = sess.run(loss_op, feed_dict={X_: X, y_: y})

    print('Iteration ', i)
    print('Predictions ', predictions)
    print('Loss ', loss)

t_end = time.clock()
print('Elapsed time ', t_end - t_start)
print("Run TensorBoard with: tensorboard --logdir=%s" % out_dir)

# Predict
sample_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
predicted = sess.run([prediction_op], 
                        feed_dict={X_: sample_X})[0]
print(sample_X)
print(predicted)