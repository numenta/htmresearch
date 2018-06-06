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

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from matlab_code.synaptic_computations import compute_hardwired_weights

# STDP kernel time constant in seconds.
SDTP_TIME_CONSTANT = 0.012


def defaultSTDPKernel(preSynActivation,
                      postSynActivation,
                      dt,
                      inhibitory = False,):
  """
  This function implements a modified version of the STDP kernel from
  Widloski & Fiete, 2014.
  :param preSynActivation: Vector of pre-synaptic activations
  :param postSynActivation: Vector of post-synaptic activations
  :param dt: the difference in time between the two (in seconds), positive if
          after and negative if before
  :return: A matrix of synapse weight changes.
  """

  stdpTimeScaler = 1
  stdpScaler = 1
  if dt < 0 and not inhibitory:
    stdpTimeScaler = 1.5
  elif dt > 0 and inhibitory:
    stdpTimeScaler = 2.
    stdpScaler = 0.5
  elif dt > 0 and not inhibitory:
    stdpTimeScaler = 2.
    stdpScaler = 1.2


  preSynActivation = np.reshape(preSynActivation, (-1, 1))
  postSynActivation = np.reshape(postSynActivation, (1, -1))
  intermediate = np.matmul(preSynActivation, postSynActivation)
  intermediate *= np.exp(dt/(SDTP_TIME_CONSTANT*-1.*stdpTimeScaler))*np.sign(dt)
  intermediate *= stdpScaler

  return intermediate




"""
This class provides a framework for learning a continuous attractor model of
a grid cell module, using rate coding.  It is loosely based on the ideas from
Widloski & Fiete, 2014, who use a similar system to learn a spiking version of
a continuous attractor network.

This class is based on having two populations of excitatory neurons, one which
is hard-wired to prefer "left" movement, and one which is hardwired to prefer
"right" movement.  It also includes a population of inhibitory neurons.
It lacks connections between excitatory neurons; all CAN dynamics are based on
inhibition.
"""

class CAN1DNetwork(object):
  def __init__(self,
               numExcitatory,
               numInhibitory,
               learningRate,
               dt,
               stdpWindow = 10,
               decayConstant = 0.03,
               velocityGain = 0.9,
               placeGainE = 10,
               placeGainI = 50,
               sigmaLoc = 0.01,
               stdpKernel = defaultSTDPKernel,
               globalTonicMagnitude = 1,
               constantTonicMagnitude = 1,
               learnFactorII = 7,
               learnFactorEI = 2,
               learnFactorIE = 1,
               envelopeWidth = 0.72,
               envelopeFactor = 60,
               initialWeightScale = 0.001):
    """

    :param numExcitatory: Size of each excitatory population.  Note that there
            are several populations, each of which has this many cells.
    :param numInhibitory: Size of inhibitory population.
    :param learningRate: The learning rate to use.
    :param dt: The time step to use for integration.
    :param stdpWindow: Number of time steps in each direction to use for STDP.
            Updates are performed only once.  This will lead to a buffer of
            length (2*sdtpWindow + 1) being created.
    :param decayConstant: The time constant for decay of neural activity
    :param velocityGain: Multiplier scaling impact of velocity.
    :param placeGainE: Multiplier scaling impact of place code on E cells.
    :param placeGainI: Multiplier scaling impact of place code on I cells.
    :param sigmaLoc: Multiplier scaling width of place code bump.
    :param stdpKernel: The STDP kernel to be used.  See the function
            defaultSTDPKernel for an example.
    :param globalTonicMagnitude: The magnitude of the global tonic input
            during training.
    :param constantTonicMagnitude: The magnitude of the non-velocity-dependent
            constant tonic input during training
    :param learnFactorII: Extra learning rate for II connections.
    :param learnFactorEI: Extra learning rate for EI connections.
    :param learnFactorIE: Extra learning rate for IE connections.
    :param envelopeWidth: The distance away from a boundary at which
             the suppressive envelope is first applied.
    :param envelopeFactor: The steepness of the suppressive envelope.
    :param initialWeightScale: The maximum initial weight value.

    """
    # Synapse weights.  We assume dense connections.
    # Inhibitory neuron recurrent weights.
    self.weightsII = np.random.random_sample((numInhibitory, numInhibitory))* \
                     initialWeightScale * -1.

    # Excitatory-to-inhibitory weights
    self.weightsELI = np.random.random_sample((numExcitatory, numInhibitory))* \
                      initialWeightScale
    self.weightsERI = np.random.random_sample((numExcitatory, numInhibitory))* \
                      initialWeightScale

    # Inhibitory-to-excitatory weights
    self.weightsIEL = np.random.random_sample((numInhibitory, numExcitatory))* \
                      initialWeightScale * -1.
    self.weightsIER = np.random.random_sample((numInhibitory, numExcitatory))* \
                      initialWeightScale * -1.

    # Determine a starting place code, which will govern activation
    # during learning.  This code is ignored during testing.
    self.placeCodeE = np.arange(0, 1, 1./numExcitatory)
    self.placeCodeI = np.arange(0, 1, 1./numInhibitory)

    self.placeGainE = placeGainE
    self.placeGainI = placeGainI
    self.velocityGain = velocityGain

    self.sigmaLoc = sigmaLoc

    self.learningRate = learningRate
    self.dt = dt
    self.decayConstant = decayConstant

    self.activationsI = np.zeros((numInhibitory,), dtype="float32")
    self.activationsER = np.zeros((numExcitatory,), dtype="float32")
    self.activationsEL = np.zeros((numExcitatory,), dtype="float32")

    self.stdpWindow = stdpWindow
    self.stdpKernel = stdpKernel

    self.activationBuffer = deque(maxlen=self.stdpWindow + 1)

    self.globalTonicMagnitude = globalTonicMagnitude
    self.constantTonicMagnitude = constantTonicMagnitude

    self.envelopeWidth = envelopeWidth
    self.envelopeFactor = envelopeFactor

    self.learnFactorII = learnFactorII
    self.learnFactorEI = learnFactorEI
    self.learnFactorIE = learnFactorIE

    self.envelopeI = self.computeEnvelope(self.placeCodeI)
    self.envelopeE = self.computeEnvelope(self.placeCodeE)



  def simulate(self, time,
               feedforwardInputI,
               feedforwardInputE,
               v,
               recurrent=True,
               dt = None):
    """
    :param time: Amount of time to simulate.
           Divided into chunks of len dt.
    :param feedforwardInputI: feedforward input to inhibitory cells.  Must have
           shape (numInhibitory,).  Should be total input over period time.
    :param feedforwardInputE: feedforward input to excitatory cells.  Must have
           shape (numExcitatory,).  Applied equally to ER and EL cells.
           Should be total input over period time.
    :param v: Velocity.  Should be a scalar.
    :param recurrent: whether or not recurrent connections should be used.
           Set to False during training to follow the methods of the original
           model.
    :return: Nothing.  All changes internal.
    """

    # Set up plotting
    self.fig = plt.figure()
    self.ax1 = self.fig.add_subplot(211)
    self.ax2 = self.fig.add_subplot(212)
    plt.ion()
    self.fig.show()
    self.fig.canvas.draw()

    self.activationsI = np.random.random_sample(self.activationsI.shape)*20
    self.activationsEL = np.random.random_sample(self.activationsEL.shape)*20
    self.activationsER = np.random.random_sample(self.activationsER.shape)*20

    if dt is None:
      oldDt = self.dt
    else:
      oldDt = self.dt
      self.dt = dt
    times = np.arange(0, time, self.dt)
    for i, t in enumerate(times):
      self.update(feedforwardInputI, feedforwardInputE, v, recurrent, True)
      if i % 10 == 0:
        self.plotActivation()

    self.dt = oldDt

  def update(self, feedforwardInputI, feedforwardInputE, v, recurrent = True,
             withEnvelope = False):
    """
    Do one update of the CAN network, of length self.dt.
    """

    deltaI = np.zeros(self.activationsI.shape)
    deltaEL = np.zeros(self.activationsEL.shape)
    deltaER = np.zeros(self.activationsER.shape)

    deltaI += feedforwardInputI
    deltaEL += feedforwardInputE
    deltaER += feedforwardInputE

    if recurrent:
      deltaI += (np.matmul(self.activationsEL, self.weightsELI) +\
                np.matmul(self.activationsER, self.weightsERI) +\
                np.matmul(self.activationsI, self.weightsII))*self.dt

      deltaEL += np.matmul(self.activationsI, self.weightsIER)*self.dt
      deltaER += np.matmul(self.activationsI, self.weightsIER)*self.dt

    deltaEL = (1 - self.velocityGain*v)*deltaEL
    deltaER = (1 + self.velocityGain*v)*deltaER

    deltaI += self.constantTonicMagnitude
    deltaEL += self.constantTonicMagnitude
    deltaER += self.constantTonicMagnitude

    if withEnvelope:
      deltaI *= self.envelopeI
      deltaER *= self.envelopeE
      deltaEL *= self.envelopeE

    deltaI -= self.activationsI/self.decayConstant
    deltaEL -= self.activationsEL/self.decayConstant
    deltaER -= self.activationsER/self.decayConstant

    deltaI *= self.dt
    deltaEL *= self.dt
    deltaER *= self.dt

    self.activationsI += deltaI
    self.activationsEL += deltaEL
    self.activationsER += deltaER

    self.activationsI = np.maximum(self.activationsI, 0., self.activationsI)
    self.activationsEL = np.maximum(self.activationsEL, 0., self.activationsEL)
    self.activationsER = np.maximum(self.activationsER, 0., self.activationsER)

  def decayWeights(self, decayConst = 60):
    self.weightsII -= self.weightsII*self.dt/decayConst
    self.weightsELI -= self.weightsELI*self.dt/decayConst
    self.weightsERI -= self.weightsERI*self.dt/decayConst
    self.weightsIEL -= self.weightsIEL*self.dt/decayConst
    self.weightsIER -= self.weightsIER*self.dt/decayConst


  def learn(self, time):
    """
    Traverses a sinusoidal trajectory across the environment, learning during
    the process.
    :param time: Amount of time, in seconds, to spend learning.
    :return: Nothing, weights updated internally.
    """

    # Set up plotting
    self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1,
                                                            gridspec_kw = {'height_ratios':[1, 1, 8],})
    plt.ion()
    self.fig.show()
    self.fig.canvas.draw()

    # Things can break if time is an int, apparently.
    time += 0.
    times = np.arange(0, time, self.dt)
    trajectory = (np.sin(times/(2) - np.pi/4)+1)/2
    velocities = np.diff(trajectory)/self.dt

    for i, t in enumerate(times[:-1]):
      x = trajectory[i]
      v = velocities[i]
      feedForwardInputI = np.exp(-1.*(self.placeCodeI - x)**2 /
                          (2*self.sigmaLoc**2))
      feedForwardInputI *= self.placeGainI
      feedForwardInputI += self.globalTonicMagnitude
      feedForwardInputE = np.exp(-1.*(self.placeCodeE - x)**2 /
                          (2*self.sigmaLoc**2))
      feedForwardInputE *= self.placeGainE
      feedForwardInputE += self.globalTonicMagnitude

      self.update(feedForwardInputI, feedForwardInputE, v, recurrent=False,
                  withEnvelope = True)
      self.stdpUpdate()
      self.decayWeights()

      if i % 10 == 0:
        self.ax3.matshow(self.weightsII)
        self.plotActivation(position = x)

    # Carry out any hanging STDP updates.
    self.stdpUpdate(clearBuffer=True)


  def normalize_weights(self, IINorm, IENorm, EINorm):
    """
    Use the L2 norm to rescale our weight matrices.
    :param IINorm: The target weight for the II weights
    :param IENorm: The target norm for both IE weight matrices
    :param EINorm: The target norm for both EI weight matrices
    :return: Nothing.  Updates done in-place
    """

    weights = [self.weightsII, self.weightsIEL, self.weightsIER,
               self.weightsELI, self.weightsERI]
    norms = [IINorm, IENorm, IENorm, EINorm, EINorm]
    for w, n in zip(weights, norms):
      norm = np.linalg.norm(w, ord = np.inf)
      w /= (norm/n)


  def computeEnvelope(self, placeCode):
    places = np.abs(placeCode - 0.5)
    envelope = [1 if p < 1 - self.envelopeWidth else
                      np.exp(-1.*self.envelopeFactor *
                      ((p - 1 + self.envelopeWidth)/self.envelopeWidth)**2)
                      for p in places]

    return np.asarray(envelope)


  def plotActivation(self, position = None):
    self.ax1.clear()
    x = np.arange(0, len(self.activationsER), 1)
    self.ax1.plot(x, self.activationsEL, color = "b", label = "EL Activation")
    self.ax1.plot(x, self.activationsER, color = "r", label = "ER Activation")
    if position is not None:
      self.ax1.axvline(x=position*len(self.activationsER))
    self.ax1.legend(loc = "best")

    self.ax2.clear()
    x = np.arange(0, len(self.activationsI), 1)
    self.ax2.plot(x, self.activationsI, color = "k", label = "I Activation")
    if position is not None:
      self.ax2.axvline(x=position*len(self.activationsI))
    self.ax2.legend(loc = "best")

    self.fig.canvas.draw()

  def stdpUpdate(self, clearBuffer=False):
    """
    Adds the current activations to the tracking queue, and then performs an
    STDP update if possible.
    :return: Nothing.  All changes made in-place.
    """
    if clearBuffer:
      while len(self.activationBuffer) > 1:
        baseI, baseEL, baseER = self.activationBuffer.popleft()
        for dt, (I, EL, ER) in enumerate(self.activationBuffer):
          t = 1. * (dt + 1) * self.dt
          self.weightsII += self.learningRate * self.stdpKernel(baseI, I, t, True)
          self.weightsIEL += self.learningRate * self.stdpKernel(baseI, EL, t, True)
          self.weightsIER += self.learningRate * self.stdpKernel(baseI, ER, t, True)
          self.weightsERI += self.learningRate * self.stdpKernel(baseER, I, t)
          self.weightsELI += self.learningRate * self.stdpKernel(baseEL, I, t)


    else:
      for dt, (I, EL, ER) in enumerate(reversed(self.activationBuffer)):
        t = -1. * (dt + 1) * self.dt
        self.weightsII +=  self.learningRate * \
                           self.stdpKernel(self.activationsI, I, t, True) * \
                           self.learnFactorII * self.dt
        self.weightsIEL += self.learningRate * \
                           self.stdpKernel(self.activationsI, EL, t, True) * \
                           self.learnFactorIE * self.dt
        self.weightsIER += self.learningRate * \
                           self.stdpKernel(self.activationsI, ER, t, True) * \
                           self.learnFactorIE * self.dt
        self.weightsERI += self.learningRate * \
                           self.stdpKernel(self.activationsEL, I, t) * \
                           self.learnFactorEI * self.dt
        self.weightsELI += self.learningRate * \
                           self.stdpKernel(self.activationsER, I, t) * \
                           self.learnFactorEI * self.dt

      for dt, (baseI, baseEL, baseER) in enumerate(self.activationBuffer):
        t = 1. * (dt + 1) * self.dt
        self.weightsII +=  self.learningRate * \
                           self.stdpKernel(baseI, self.activationsI, t, True) * \
                           self.learnFactorII * self.dt
        self.weightsIEL += self.learningRate * \
                           self.stdpKernel(baseI, self.activationsEL, t, True) * \
                           self.learnFactorIE * self.dt
        self.weightsIER += self.learningRate * \
                           self.stdpKernel(baseI, self.activationsEL, t, True) * \
                           self.learnFactorIE * self.dt
        self.weightsERI += self.learningRate * \
                           self.stdpKernel(baseER, self.activationsI, t) * \
                           self.learnFactorEI * self.dt
        self.weightsELI += self.learningRate * \
                           self.stdpKernel(baseEL, self.activationsI, t) * \
                           self.learnFactorEI * self.dt

      self.activationBuffer.append((np.copy(self.activationsI),
                                    np.copy(self.activationsEL),
                                    np.copy(self.activationsER)))

if __name__ == "__main__":
  network = CAN1DNetwork(200, 80, .015, .001, decayConstant=0.1)
  network.learn(1)
  import ipdb; ipdb.set_trace()
  sns.distplot(network.activationsI)
  plt.show()




