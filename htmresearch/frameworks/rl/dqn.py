# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
import torch
import torch.nn.functional as F
import torch.optim as optim
import collections
import random
import copy

class DQN(object):
  """
  Implements **DQN algorithm** described in https://arxiv.org/pdf/1312.5602.pdf
  """


  def __init__(self, actions, network,
               eps_start=1.0, eps_end=0.01, eps_decay=0.9995,
               learning_rate=0.0001, gamma=0.99, tau=1.0, target_update=1000,
               batch_size=32, min_steps=10000, replay_size=10000):
    """
    :param actions: Number of possible actions
    :param network: Neural neural network to use as a Q function approximator
    :param eps_start: e-greedy exploration start epsilon
    :param eps_end: e-greedy exploration end epsilon
    :param eps_decay: e-greedy exploration decay
    :param learning_rate: optimizer learning rate
    :param gamma: future returns discount
    :param tau: soft update smoothing coefficient. Use 1.0 for hard target update
    :param target_update: target update interval
    :param batch_size: batch size
    :param min_steps: min number of experiences in replay buffer before learning
    :param replay_size: replay memory size
    """

    self.actions = actions

    # e-greedy parameters
    self.eps = eps_start
    self.eps_end = eps_end
    self.eps_decay = eps_decay

    # Experience replay memory: e = (s, a, r, s', done)
    self.replay = collections.deque(maxlen=replay_size)
    self.min_steps = min_steps
    self.batch_size = batch_size

    # Initialize Local and Target networks
    self.local = network
    self.target = copy.deepcopy(self.local)
    self.device = next(network.parameters()).device

    # Optimizations
    self.optimizer = optim.Adam(self.local.parameters(), lr=learning_rate)
    self.gamma = gamma
    self.tau = tau
    self.target_update = target_update
    self.steps = 0


  def select_action(self, state):
    """
    Select the best action for the given state using e-greedy exploration to
    minimize overfitting

    :return: tuple(action, value)
    """
    value = 0
    if self.steps < self.min_steps:
      action = np.random.randint(self.actions)
    else:
      self.eps = max(self.eps_end, self.eps * self.eps_decay)
      if random.random() < self.eps:
        action = np.random.randint(self.actions)
      else:
        self.local.eval()
        with torch.no_grad():
          state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
          Q = self.local(state)
          value, action = torch.max(Q, 1)

    return int(action), float(value)


  def hard_target_update(self):
    """
    Update model parameters every 'target_update' time steps
    See https://arxiv.org/abs/1312.5602
    :param target_update: target update interval
    """
    if self.steps % self.target_update == 0:
      self.target.load_state_dict(self.local.state_dict())


  def soft_target_update(self):
    """
    Soft update model parameters:

    .. math::
      \\theta_target = \\tau \\times \\theta_local + (1 - \\tau) \\times \\theta_target ,
      with \\tau \\ll 1

    See https://arxiv.org/pdf/1509.02971.pdf
    """
    for target_param, local_param in zip(self.target.parameters(), self.local.parameters()):
      target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


  def learn(self, state, action, reward, next_state, done):
    """
    Update replay memory and learn from a batch of random experiences sampled
    from the replay buffer
    :return: optimization loss if enough experiences are available, None otherwise
    """
    self.steps += 1
    self.replay.append((state, action, reward, next_state, done))
    if self.steps > self.min_steps and len(self.replay) > self.batch_size:
      batch = random.sample(self.replay, self.batch_size)
      return self.optimize(batch)

    return None


  def optimize(self, batch):
    state, action, reward, next_state, done = map(
      lambda x: torch.tensor(x, device=self.device, dtype=torch.float), zip(*batch))

    # Get target values
    self.target.eval()
    with torch.no_grad():
      target_next = self.target(next_state)
      target_next = torch.max(target_next, dim=1, keepdim=True)[0]
    Q_target = reward + self.gamma * target_next * (1 - done)

    # Get actual values
    self.local.train()
    Q_local = self.local(state)
    Q_local = Q_local.index_select(1, action.long())

    # Compute Huber loss
    loss = F.smooth_l1_loss(Q_local, Q_target)

    self.optimizer.zero_grad()
    loss.backward()
    for param in self.local.parameters():
      param.grad.data.clamp_(-1, 1)

    self.optimizer.step()

    # Update target model
    if self.tau < 1:
      self.soft_target_update()
    else:
      self.hard_target_update()

    return loss.item()

