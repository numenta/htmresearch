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
"""
OpenAI Atari utilities extracted from https://github.com/openai/baselines

see https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
from collections import deque

import gym
import numpy as np
from torchvision import transforms



class NoopResetEnv(gym.Wrapper):
  def __init__(self, env, noop_max=30):
    """Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """
    gym.Wrapper.__init__(self, env)
    self.noop_max = noop_max
    self.override_num_noops = None
    self.noop_action = 0
    assert env.unwrapped.get_action_meanings()[0] == 'NOOP'


  def reset(self, **kwargs):
    """ Do no-op action for a number of steps in [1, noop_max]."""
    self.env.reset(**kwargs)
    if self.override_num_noops is not None:
      noops = self.override_num_noops
    else:
      noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
    assert noops > 0
    obs = None
    for _ in range(noops):
      obs, _, done, _ = self.env.step(self.noop_action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs


  def step(self, ac):
    return self.env.step(ac)



class FireResetEnv(gym.Wrapper):
  def __init__(self, env):
    """Take action on reset for environments that are fixed until firing."""
    gym.Wrapper.__init__(self, env)
    assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
    assert len(env.unwrapped.get_action_meanings()) >= 3


  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(1)
    if done:
      self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(2)
    if done:
      self.env.reset(**kwargs)
    return obs


  def step(self, ac):
    return self.env.step(ac)



class EpisodicLifeEnv(gym.Wrapper):
  def __init__(self, env):
    """Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """
    gym.Wrapper.__init__(self, env)
    self.lives = 0
    self.was_real_done = True


  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.was_real_done = done
    # check current lives, make loss of life terminal,
    # then update lives to handle bonus lives
    lives = self.env.unwrapped.ale.lives()
    if lives < self.lives and lives > 0:
      # for Qbert sometimes we stay in lives == 0 condition for a few frames
      # so it's important to keep lives > 0, so that we only reset once
      # the environment advertises done.
      done = True
    self.lives = lives
    return obs, reward, done, info


  def reset(self, **kwargs):
    """Reset only when lives are exhausted.
    This way all states are still reachable even though lives are episodic,
    and the learner need not know about any of this behind-the-scenes.
    """
    if self.was_real_done:
      obs = self.env.reset(**kwargs)
    else:
      # no-op step to advance from terminal/lost life state
      obs, _, _, _ = self.env.step(0)
    self.lives = self.env.unwrapped.ale.lives()
    return obs



class ClipRewardEnv(gym.RewardWrapper):
  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)


  def reward(self, reward):
    """Bin reward to {+1, 0, -1} by its sign."""
    return np.sign(reward)



class WarpFrame(gym.ObservationWrapper):
  """
  Warp frames to 84x84 as done in the Nature paper and later work.
  """


  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    self.transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Grayscale(),
                                         transforms.Resize(size=(110, 84)),
                                         transforms.CenterCrop(84),
                                         transforms.ToTensor()])


  def observation(self, frame):
    return self.transform(frame)



class FrameStack(gym.Wrapper):
  def __init__(self, env, k):
    """
    Stack k last frames.
    """
    gym.Wrapper.__init__(self, env)
    self.frames = deque(maxlen=k)
    self.k = k


  def reset(self):
    reset = self.env.reset()
    self.frames.extend([reset] * self.k)
    return np.concatenate(self.frames, axis=0)


  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.frames.append(obs)
    obs = np.concatenate(self.frames, axis=0)
    return obs, reward, done, info



def create_atari_environment(name):
  """
  Create OpenAI Gym atari environment
  :param name: name of  the atari game
  :return: Preconfigured environment suitable for atari games
  """
  env = gym.make(name)
  env = NoopResetEnv(env, noop_max=30)
  env = EpisodicLifeEnv(env)
  env = ClipRewardEnv(env)

  action_names = env.unwrapped.get_action_meanings()
  if 'FIRE' in action_names:
    env = FireResetEnv(env)

  env = WarpFrame(env)
  env = FrameStack(env, k=4)
  return env
