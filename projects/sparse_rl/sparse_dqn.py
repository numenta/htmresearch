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

from __future__ import print_function

import argparse
import collections
import copy
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from htmresearch.frameworks.pytorch.sparse_net import SparseNet
from htmresearch.support.openai_utils import create_atari_environment

SEED = 42
SPARSE = True
NUM_TIMESTEPS = int(1e6)
ENVIRONMENT = "SeaquestDeterministic-v4"
RESULTS_PATH = "results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Flatten(nn.Module):
  """
  Flatten input retaining batch dimension
  """


  def forward(self, x):
    return x.view(x.size(0), -1)



class DenseQ(nn.Module):
  """
  Neural Network described in https://arxiv.org/pdf/1312.5602.pdf
  """


  def __init__(self, actions):
    super(DenseQ, self).__init__()

    inputShape = (4, 84, 84)
    self.model = nn.Sequential(
      nn.Conv2d(in_channels=inputShape[0], out_channels=16, kernel_size=8,
                stride=4),
      nn.ReLU(),
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
      nn.ReLU(),
      Flatten(),
      nn.Linear(in_features=32 * 9 * 9, out_features=256),
      nn.LayerNorm(256),
      nn.ReLU(),
      nn.Linear(in_features=256, out_features=actions),
    )


  def forward(self, x):
    return self.model(x)



class SparseQ(nn.Module):
  """
  Sparse version of DenseQ network
  """


  def __init__(self, actions):
    super(SparseQ, self).__init__()

    self.model = SparseNet(inputSize=(4, 84, 84),
                           outChannels=[16, 32],
                           kernelSize=[8, 4],
                           stride=[4, 2],
                           n=256,
                           outputSize=actions,
                           c_k=[160, 320], k=40, weightSparsity=0.3,
                           useBatchNorm=True,
                           useSoftmax=False)


  def forward(self, x):
    return self.model(x)



class DQN(object):
  """
  Implements **DQN algorithm** described in https://arxiv.org/pdf/1312.5602.pdf
  """


  def __init__(self, environment, network,
               eps_start=1.0, eps_end=0.01, eps_decay=0.9995,
               learning_rate=0.0001, gamma=0.99, tau=1.0, target_update=1000,
               batch_size=32, min_steps=10000, replay_size=10000):
    """
    :param environment: Open AI Gym environment
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

    self.env = environment

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
    self.optimizer = optim.Adam(self.local.parameters(), lr=learning_rate)

    # Optimizations
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
      action = self.env.action_space.sample()
    else:
      self.eps = max(self.eps_end, self.eps * self.eps_decay)
      if random.random() < self.eps:
        action = self.env.action_space.sample()
      else:
        self.local.eval()
        with torch.no_grad():
          state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
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
      lambda x: torch.tensor(x, device=device, dtype=torch.float), zip(*batch))

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



def run(env, model, learn, numSteps, render, resultsPath, checkpoint=None):
  """
  Main RL loop where the agent acts on the environment based on the states
  :param environment: Open AI Gym environment
  :param model: Neural neural network model to use as a Q function approximator
  :param learn: Controls learning. Set to False when using a pre-trained model
  :param numSteps: Total number of steps/frames to run
  :param render: Whether or not to render the environment as it runs
  :param resultsPath: Where to store the results
  :param checkpoint: How often should the model be saved. None for never
  """

  path = os.path.join(resultsPath, "{0}.{1}.csv".format(
    env.spec.id, "train" if learn else "eval"))
  with open(path, "w") as f:
    log = csv.writer(f)
    log.writerow(["episode", "score", "losses", "values"])
    score = 0
    episode = 0
    loss = 0

    # Train for numSteps
    agent = DQN(env, model)
    state = env.reset()
    progess_bar = tqdm.trange(numSteps, desc="t")
    for t in progess_bar:
      if render:
        env.render()

      action, value = agent.select_action(state)
      next_state, reward, done, info = env.step(action)
      if learn:
        loss = agent.learn(state, action, reward, next_state, done)
      state = next_state

      # Collect stats
      score += reward
      log.writerow([episode, score, loss, value])

      # Save checkpoint
      if checkpoint is not None and t % checkpoint == 0:
        path = os.path.join(resultsPath, "{0}.checkpoint.{1:08d}".format(env.spec.id, t))
        torch.save(model, path)

      if value > 0:
        progess_bar.set_description("{0}:{1}:{2}".format(episode, score, value))

      if done:
        episode += 1
        state = env.reset()
        score = 0

  if learn:
    path = os.path.join(resultsPath, "{0}.pt".format(env.spec.id))
    torch.save(model, path)



def main(opts):
  try:
    os.mkdir(opts.results_path)
  except OSError:
    pass

  random.seed(opts.seed)
  np.random.seed(opts.seed)
  torch.manual_seed(opts.seed)

  # Create environment
  env = create_atari_environment(opts.env)
  env.seed(opts.seed)

  # Look for trained model
  path = os.path.join(opts.results_path, "{0}.pt".format(env.spec.id))
  if os.path.exists(path):
    learn = False
    model = torch.load(path, map_location=device)
  else:
    learn = True
    if opts.sparse:
      model = SparseQ(actions=env.action_space.n).to(device)
    else:
      model = DenseQ(actions=env.action_space.n).to(device)

  run(env=env, model=model, learn=learn, numSteps=opts.timesteps,
      render=opts.render, resultsPath=opts.results_path)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--env", default=ENVIRONMENT, metavar="STRING",
                      type=str, help="OpenAI Gym atari environment name")
  parser.add_argument("--sparse", action="store_true", default=SPARSE)
  parser.add_argument("--dense", action="store_false", dest="sparse")
  parser.add_argument("--timesteps", default=NUM_TIMESTEPS, metavar="N",
                      type=int, help="Number of time steps to run")
  parser.add_argument("--results_path", default=RESULTS_PATH, metavar="PATH",
                      type=str, help="Path to store results")
  parser.add_argument("--seed", default=SEED, type=int,
                      help="Random seed to use")
  parser.add_argument("--render", action="store_true", default=False,
                      help="Render environment UI")

  opts = parser.parse_args()
  for arg in vars(opts):
    print(arg, "=", getattr(opts, arg))

  main(opts)
