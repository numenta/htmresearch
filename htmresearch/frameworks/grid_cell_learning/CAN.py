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
import copy
import os
from compute_hardwired_weights import compute_hardwired_weights

# STDP kernel time constant in seconds.  Used for the default kernel.
SDTP_TIME_CONSTANT = 0.012

# How often to update plots
PLOT_INTERVAL = 0.1

# How often path integration estimates are collected.  This needs to be tuned,
# as values that are too small will lead to constant estimates of zero movement.
ESTIMATION_INTERVAL = 0.5

def defaultSTDPKernel(preSynActivation,
                      postSynActivation,
                      dt,
                      inhibitoryPresyn=False,
                      inhibitoryPostsyn=False):
  """
  This function implements a modified version of the STDP kernel from
  Widloski & Fiete, 2014.
  :param preSynActivation: Vector of pre-synaptic activations
  :param postSynActivation: Vector of post-synaptic activations
  :param dt: the difference in time between the two (in seconds), positive if
          after and negative if before
  :return: A matrix of synapse weight changes.
  """

  stdpScaler = 1
  stdpTimeScaler = 1.

  # Set up STDP directions
  if inhibitoryPresyn and not inhibitoryPostsyn:
    #I-E, anti-Hebbian (weakening inhibitory connections)
    stdpScaler *= 1
  elif not inhibitoryPresyn and inhibitoryPostsyn:
    # E-I, anti-Hebbian
    stdpScaler *= -1
  elif inhibitoryPresyn and inhibitoryPostsyn:
    # I-I, Hebbian (strengthening inhibitory connections)
    stdpScaler *= -1

  # Set up parameters
  if dt < 0 and not inhibitoryPresyn:
    # Anti-causal
    stdpScaler *= 1
    stdpTimeScaler *= 3
  elif dt > 0 and not inhibitoryPresyn:
    stdpScaler *= 1.2
    stdpTimeScaler *= 4
  elif dt > 0 and inhibitoryPresyn:
    stdpScaler *= .5
    stdpTimeScaler *= 4
  elif dt < 0 and inhibitoryPresyn:
    stdpScaler *= 1
    stdpTimeScaler *= 2

  timeFactor = np.exp(-1*np.abs(dt)/(SDTP_TIME_CONSTANT*stdpTimeScaler))
  preSynActivation *= timeFactor*np.sign(dt)*stdpScaler

  updates = np.outer(preSynActivation, postSynActivation)

  return updates


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
               stdpWindow=10,
               decayConstant=0.03,
               velocityGain=0.9,
               placeGainE=10,
               placeGainI=50,
               sigmaLoc=0.01,
               stdpKernel=defaultSTDPKernel,
               globalTonicMagnitude=0,
               constantTonicMagnitude=0,
               learnFactorII=7,
               learnFactorEI=2,
               learnFactorIE=1,
               envelopeWidth=0.8,
               envelopeFactor=25,
               initialWeightScale=0.003,
               clip=10,
               plotting=True):
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
    :param clip: The maximum possible activation.
    :param plotting: Whether or not to generate plots.  False speeds training.

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

    self.activationBuffer = deque(maxlen=int(self.stdpWindow))

    self.globalTonicMagnitude = globalTonicMagnitude
    self.constantTonicMagnitude = constantTonicMagnitude

    self.envelopeWidth = envelopeWidth
    self.envelopeFactor = envelopeFactor

    self.learnFactorII = learnFactorII
    self.learnFactorEI = learnFactorEI
    self.learnFactorIE = learnFactorIE

    self.envelopeI = self.computeEnvelope(self.placeCodeI)
    self.envelopeE = self.computeEnvelope(self.placeCodeE)

    self.clip = clip
    self.plotting = plotting



  def calculatePathIntegrationError(self, time, dt=None, trajectory=None,
                                    envelope=False, inputNoise=None):
    """
    Calculate the error of our path integration, relative to an ideal module.
    To do this, we track the movement of an individual bump

    Note that the network must be trained before this is done.
    :param time: How long to simulate for in seconds.  We recommend using a
            small value, e.g. ~10s.
    :param trajectory: An optional trajectory that specifies how the network moves.
    :param inputNoise: Whether or not to apply noise, and how much.
    :return: A tuple of the true trajectory and the inferred trajectory.
    """
    # Set up plotting
    if self.plotting:
      self.fig = plt.figure()
      self.ax1 = self.fig.add_subplot(411)
      self.ax2 = self.fig.add_subplot(412)
      self.ax3 = self.fig.add_subplot(413)
      self.ax4 = self.fig.add_subplot(414)
      plt.tight_layout()
      plt.ion()
      self.fig.show()
      self.fig.canvas.draw()
      mouse = plt.imread(os.path.dirname(os.path.realpath(__file__))
                         + "/mouse_graphic.png")

      self.ax1.set_xlabel("Excitatory population activity")
      self.ax2.set_xlabel("Inhibitory population activity")
      self.ax3.set_xlabel("Movement in cells")
      self.ax3.set_ylabel("Cost")
      self.ax4.set_xlabel("Location")

      plt.tight_layout()


    if dt is None:
      oldDt = self.dt
    else:
      oldDt = self.dt
      self.dt = dt

    # Simulate for a second to get nice starting activation bumps.
    # Turn plotting off so as not to confuse the viewer
    oldPlotting = self.plotting
    self.plotting = False
    self.simulate(1, 1, 1, 0, envelope=envelope, inputNoise=None)
    self.plotting = oldPlotting


    estimatedVelocities = []
    trueVelocities = []

    times = np.arange(0, time, self.dt)
    if trajectory is None:
      # Sum together two different sinusoidals for a more interesting path.
      trajectory = (np.sin((-times*np.pi/10 - np.pi/2.))+1)*2.5
      trajectory += (np.cos((-times*np.pi/3 - np.pi/2.))+1)*.75
      velocities = np.diff(trajectory)/self.dt

    oldActivations = copy.copy(self.activationsI)
    oldX = trajectory[0]
    for i, t in enumerate(times[:-1]):
      v = velocities[i]
      x = trajectory[i]

      feedforwardInputI = np.ones(self.activationsI.shape)
      feedforwardInputE = np.ones(self.activationsEL.shape)

      if inputNoise is not None:
        noisesI = np.random.random_sample(feedforwardInputI.shape)*inputNoise
        noisesE = np.random.random_sample(feedforwardInputE.shape)*inputNoise
      else:
        noisesE = 1.
        noisesI = 1.

      self.update(feedforwardInputI*noisesI, feedforwardInputE*noisesE,
                  v, True, envelope=envelope)

      estimationTime = np.abs(np.mod(t, ESTIMATION_INTERVAL))
      if estimationTime <= 0.00001 or \
          np.abs(estimationTime - ESTIMATION_INTERVAL) <= 0.00001:

          rotations = [np.sum(np.abs(np.roll(oldActivations, i) -
                                     self.activationsI))
                       for i in range(-20, 21, 1)]

          shift = np.argmin(rotations) - 20

          trueVelocities.append(x - oldX)
          oldX = x
          oldActivations = copy.copy(self.activationsI)
          estimatedVelocities.append(shift)

      if self.plotting:
        plotTime = np.abs(np.mod(t, PLOT_INTERVAL))
        if plotTime <= 0.00001 or np.abs(plotTime - PLOT_INTERVAL) <= 0.00001:
          self.ax3.clear()
          self.ax3.plot(np.arange(-len(rotations)/2 + 1, len(rotations)/2 + 1, 1),
                                  rotations,
                                  color="g",
                                  label="Shift")
          self.ax3.legend(loc="best")
          self.ax3.set_xlabel("Movement in cells")
          self.ax3.set_ylabel("Cost")
          self.ax3.axvline(x=shift)

          self.ax4.clear()
          self.ax4.set_xlim(np.amin(trajectory), np.amax(trajectory))
          self.ax4.set_ylim(0, 1)
          mouse_bound = (x - 0.25*np.sign(v), x + 0.25*np.sign(v), .05, .55)
          self.ax4.imshow(mouse,
                          aspect='auto',
                          extent=mouse_bound,
                          zorder=-1)
          self.ax4.set_xlabel("Location")
          self.ax4.axes.get_yaxis().set_visible(False)

          self.fig.canvas.draw()

          self.plotActivation(time=t, velocity=v)

    self.dt = oldDt
    return(np.asarray(trueVelocities), np.asarray(estimatedVelocities))


  def hardwireWeights(self, flip=False):
    (G_I_EL, G_I_ER, G_EL_I, G_ER_I, G_I_I) = \
      compute_hardwired_weights(2.2,
                                self.activationsEL.shape[0],
                                self.activationsI.shape[0],
                                True)

    # We need to flip the signs for the inhibitory weights;
    # in our convention, inhibitory weights are always negative,
    # but in theirs, they are positive and the sign flip is applied
    # during activation.
    self.weightsII = -1.*G_I_I

    # If we want the network to path integrate in the right direction,
    # flip ELI and ERI.
    if flip:
      self.weightsELI = G_ER_I
      self.weightsERI = G_EL_I
      self.weightsIEL = -1.*G_I_ER
      self.weightsIER = -1.*G_I_EL

    else:
      self.weightsELI = G_EL_I
      self.weightsERI = G_ER_I
      self.weightsIEL = -1. * G_I_EL
      self.weightsIER = -1. * G_I_ER


  def simulate(self, time,
               feedforwardInputI,
               feedforwardInputE,
               v,
               recurrent=True,
               dt=None,
               envelope=False,
               inputNoise=None):
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
    if self.plotting:
      self.fig = plt.figure()
      self.ax1 = self.fig.add_subplot(211)
      self.ax2 = self.fig.add_subplot(212)
      plt.ion()

      self.ax1.set_xlabel("Excitatory population activity")
      self.ax2.set_xlabel("Inhibitory population activity")

      plt.tight_layout()
      self.fig.show()
      self.fig.canvas.draw()

    self.activationsI = np.random.random_sample(self.activationsI.shape)
    self.activationsEL = np.random.random_sample(self.activationsEL.shape)
    self.activationsER = np.random.random_sample(self.activationsER.shape)

    if dt is None:
      oldDt = self.dt
    else:
      oldDt = self.dt
      self.dt = dt
    times = np.arange(0, time, self.dt)
    for i, t in enumerate(times):
      if inputNoise is not None:
        noisesI = np.random.random_sample(feedforwardInputI.shape)*inputNoise
        noisesE = np.random.random_sample(feedforwardInputE.shape)*inputNoise
      else:
        noisesE = 1.; noisesI = 1.

      self.update(feedforwardInputI*noisesI, feedforwardInputE*noisesE,
                  v, recurrent, envelope=envelope)
      if self.plotting:
        plotTime = np.abs(np.mod(t, PLOT_INTERVAL))
        if plotTime <= 0.00001 or np.abs(plotTime - PLOT_INTERVAL) <= 0.00001:
          self.plotActivation(time=t, velocity=v)

    self.dt = oldDt

  def update(self, feedforwardInputI, feedforwardInputE, v, recurrent=True,
             envelope=False, iSpeedTuning=False, enforceDale=True):
    """
    Do one update of the CAN network, of length self.dt.
    :param feedforwardInputI: The feedforward input to inhibitory cells.
    :param feedforwardInputR: The feedforward input to excitatory cells.
    :param v: The current velocity.
    :param recurrent: Whether or not recurrent connections should be used.
    :param envelope: Whether or not an envelope should be applied.
    :param iSpeedTuning: Whether or not inhibitory cells should also have their
             activations partially depend on current movement speed.  This is
             necessary for periodic training, serving a role similar to that of
             the envelope.
    :param Whether or not Dale's law should be enforced locally.  Helps with
             training with recurrent weights active, but can slow down training.
    """

    deltaI = np.zeros(self.activationsI.shape)
    deltaEL = np.zeros(self.activationsEL.shape)
    deltaER = np.zeros(self.activationsER.shape)

    deltaI += feedforwardInputI
    deltaEL += feedforwardInputE
    deltaER += feedforwardInputE

    if enforceDale:
      weightsII =  np.minimum(self.weightsII, 0)
      weightsIER = np.minimum(self.weightsIER, 0)
      weightsIEL = np.minimum(self.weightsIEL, 0)
      weightsELI = np.maximum(self.weightsELI, 0)
      weightsERI = np.maximum(self.weightsERI, 0)
    else:
      weightsII =  self.weightsII
      weightsIER = self.weightsIER
      weightsIEL = self.weightsIEL
      weightsELI = self.weightsELI
      weightsERI = self.weightsERI

    if recurrent:
      deltaI += (np.matmul(self.activationsEL, weightsELI) +\
                np.matmul(self.activationsER, weightsERI) +\
                np.matmul(self.activationsI, weightsII))

      deltaEL += np.matmul(self.activationsI, weightsIEL)
      deltaER += np.matmul(self.activationsI, weightsIER)

    deltaEL *= max((1 - self.velocityGain*v), 0)
    deltaER *= max((1 + self.velocityGain*v), 0)
    if iSpeedTuning:
      deltaI *= min(self.velocityGain*np.abs(v), 1)

    deltaI += self.constantTonicMagnitude
    deltaEL += self.constantTonicMagnitude
    deltaER += self.constantTonicMagnitude

    if envelope:
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

    # Activations by definition must be positive
    np.maximum(self.activationsI, 0., self.activationsI)
    np.maximum(self.activationsEL, 0., self.activationsEL)
    np.maximum(self.activationsER, 0., self.activationsER)

    # Clip activations for stability
    np.minimum(self.activationsI, self.clip, self.activationsI)
    np.minimum(self.activationsEL, self.clip, self.activationsEL)
    np.minimum(self.activationsER, self.clip, self.activationsER)


  def decayWeights(self, decayConst=60):
    """
    Decay the network's weights.

    :param decayConst: The time constant (in seconds) to use for decay.
            Note: If applied, decay must be used extremely carefully, as
            it has a tendency to cause asymmetries in the network weights.
    """
    self.weightsII -= self.weightsII*self.dt/decayConst
    self.weightsELI -= self.weightsELI*self.dt/decayConst
    self.weightsERI -= self.weightsERI*self.dt/decayConst
    self.weightsIEL -= self.weightsIEL*self.dt/decayConst
    self.weightsIER -= self.weightsIER*self.dt/decayConst


  def learn(self,
            runs,
            dir=1,
            periodic=False,
            recurrent=True,
            randomSpeed=False):
    """
    Traverses a sinusoidal trajectory across the environment, learning during
    the process.  A pair of runs across the environment (one in each direction)
    takes 10 seconds if in a periodic larger environment, and 4 seconds in a 
    smaller nonperiodic environment.
    :param runs: How many runs across the environment to do.  Each "run" is
            defined as a full sweep across the environment in each direction.
    :param dir: Which direction to move in first.  Valid values are 1 and -1.
    :param periodic: Whether or not the learning environment should be
            periodic (toroidal).
    :param recurrent: Whether or not recurrent connections should be active
            during learning.  Warning: True can lead to instability.
    :param randomSpeed: Whether or not to use a random maximum speed for each
            run, to better simulate real learning.  Can degrade performance.
            Only supported in periodic environments.
    """
    # Set up plotting
    if self.plotting:
      self.fig = plt.figure()
      self.ax1 = self.fig.add_subplot(411)
      self.ax2 = self.fig.add_subplot(412)
      self.ax3 = self.fig.add_subplot(212)

      plt.ion()
      plt.tight_layout()
      self.ax3.set_xlabel("Inhibitory-Inhibitory connections")
      self.fig.show()
      self.fig.canvas.draw()

    # Set up the trajectories and running times.
    if not periodic:
      time = 4.*runs
      timings = [np.arange(0, time, self.dt)]
      trajectories = [(np.sin(dir*(times*np.pi/2 - np.pi/2.))+1)/2]
    else:
      # Space the starting points of the runs out.  This tends to improve the
      # translation-invariance of the weight profiles, and thus gives better
      # overall path integration.
      startingPoint = 0
      trajectories = []
      timings = []
      time = 0
      residTime = 0
      for run in xrange(runs):
        if randomSpeed:
          speed = np.random.random() + 0.5
        else:
          speed = 1.
        length = 10./speed

        runTimes = np.arange(0, length, self.dt)
        trajectory = (np.sin(dir*(runTimes*np.pi/(5/speed) - np.pi/2.)) + 1)*\
                      2.5 + startingPoint
        trajectories.append(trajectory)
        timings.append(runTimes + time)
        time += length
        startingPoint += 1./runs



    for trajectory, timing in zip(trajectories, timings):
      self.activationsI = np.zeros(self.activationsI.shape)
      self.activationsER = np.zeros(self.activationsER.shape)
      self.activationsEL = np.zeros(self.activationsEL.shape)
      velocities = np.diff(trajectory)/self.dt
      for i, t in enumerate(timing[:-1]):
        x = trajectory[i] % 1
        v = velocities[i]
        feedforwardInputI = np.exp(-1.*(self.placeCodeI - x)**2 /
                            (2*self.sigmaLoc**2))
        feedforwardInputI *= self.placeGainI
        feedforwardInputI += self.globalTonicMagnitude
        feedforwardInputE = np.exp(-1.*(self.placeCodeE - x)**2 /
                            (2*self.sigmaLoc**2))
        feedforwardInputE *= self.placeGainE
        feedforwardInputE += self.globalTonicMagnitude

        self.update(feedforwardInputI,
                    feedforwardInputE,
                    v,
                    recurrent=recurrent,
                    envelope=(not periodic),
                    iSpeedTuning=periodic,
                    enforceDale=True,
                    )
        self.stdpUpdate(time=i)

        if self.plotting:
          residTime += self.dt
          if residTime > PLOT_INTERVAL:
            residTime -= PLOT_INTERVAL
            self.ax3.matshow(self.weightsII, cmap=plt.cm.coolwarm)
            self.plotActivation(position=x, time=t)

      # Carry out any hanging STDP updates.
      self.stdpUpdate(time=i, clearBuffer=True)

    # Finally, enforce Dale's law.  Inhibitory neurons must be inhibitory,
    # excitatory neurons must be excitatory.
    # This could be handled through update, but it's faster to do it here.
    np.minimum(self.weightsII, 0, self.weightsII)
    np.minimum(self.weightsIER, 0, self.weightsIER)
    np.minimum(self.weightsIEL, 0, self.weightsIEL)
    np.maximum(self.weightsELI, 0, self.weightsELI)
    np.maximum(self.weightsERI, 0, self.weightsERI)


  def normalize_weights(self, IIMax, IEMax, EIMax):
    """
    Rescale our weight matrices to have a certain maximum absolute value.
    :param IINorm: The target maximum for the II weights
    :param IENorm: The target maximum for both IE weight matrices
    :param EINorm: The target maximum for both EI weight matrices
    """
    weights = [self.weightsII, self.weightsIEL, self.weightsIER,
               self.weightsELI, self.weightsERI]
    norms = [IIMax, IEMax, IEMax, EIMax, EIMax]
    for w, n in zip(weights, norms):
      maximum = np.amax(np.abs(w))
      w /= maximum
      w *= n


  def computeEnvelope(self, placeCode):
    """
    Compute an envelope for use in suppressing border cells.
    :param placeCode: The place code representing the population the envelope
            will be used for.
    :return: A numpy array that can be elementwise-multiplied with activations
             for the given cell population to apply the envelope.
    """
    places = np.abs(placeCode - 0.5)
    envelope = [1 if p < 1 - self.envelopeWidth else
                      np.exp(-1.*self.envelopeFactor *
                      ((p - 1 + self.envelopeWidth)/self.envelopeWidth)**2)
                      for p in places]

    return np.asarray(envelope)


  def plotActivation(self, position=None, time=None, velocity=None):
    """
    Plot the activation of the current cell populations.  Assumes that
    two axes have already been created, ax1 and ax2.  If done in a Jupyter
    notebook, this plotting will overwrite the old plot.
    :param position: The current location of the animal
    :param time: The current time in the simulation
    :param velocity: The current velocity of the animal
    """
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

    titleString = ""
    if time is not None:
      titleString += "Time = {}".format(str(time))
    if velocity is not None:
      titleString += "  Velocity = {}".format(str(velocity)[:4])
    if position is not None:
      titleString += "  Position = {}".format(str(position)[:4])
    self.ax1.set_title(titleString)

    self.ax1.set_xlabel("Excitatory activity")
    self.ax2.set_xlabel("Inhibitory activity")

    self.fig.canvas.draw()

  def stdpUpdate(self, time, clearBuffer=False):
    """
    Adds the current activations to the tracking queue, and then performs an
    STDP update if possible.
    :param time: The current time.  Must be provided.
    :param clearBuffer: Set as True to clear the activation buffer.
            This should be done at the end of training.
    """
    if clearBuffer:
      while len(self.activationBuffer) > 1:
        baseI, baseEL, baseER, t = self.activationBuffer.popleft()
        for (I, EL, ER, i) in self.activationBuffer:
          t = 1. * (i - t) * self.dt
          self.weightsII += self.stdpKernel(self.learningRate *\
                                            self.learnFactorII *\
                                            self.dt *\
                                            self.activationsI, I, t,
                                            True, True)

          self.weightsIEL += self.stdpKernel(self.learningRate *\
                                             self.learnFactorIE *\
                                             self.dt *\
                                             self.activationsI, EL, t,
                                             True, False)

          self.weightsIER += self.stdpKernel(self.learningRate *\
                                             self.learnFactorIE *\
                                             self.dt *\
                                             self.activationsI, ER, t,
                                             True, False)

          self.weightsELI += self.stdpKernel(self.learningRate *\
                                             self.learnFactorEI *\
                                             self.dt *\
                                             self.activationsEL, I, t,
                                             False, True)

          self.weightsERI += self.stdpKernel(self.learningRate *\
                                             self.learnFactorEI *\
                                             self.dt *\
                                             self.activationsER, I, t,
                                             False, True)

    else:
      for I, EL, ER, i in reversed(self.activationBuffer):
        t = (i - time) * self.dt
        self.weightsII +=  self.stdpKernel(self.learningRate *\
                                           self.learnFactorII *\
                                           self.dt *\
                                           self.activationsI, I, t,
                                           True, True)

        self.weightsIEL += self.stdpKernel(self.learningRate *\
                                           self.learnFactorIE *\
                                           self.dt *\
                                           self.activationsI, EL, t,
                                           True, False)

        self.weightsIER += self.stdpKernel(self.learningRate *\
                                           self.learnFactorIE *\
                                           self.dt *\
                                           self.activationsI, ER, t,
                                           True, False)

        self.weightsELI += self.stdpKernel(self.learningRate *\
                                           self.learnFactorEI *\
                                           self.dt *\
                                           self.activationsEL, I, t,
                                           False, True)

        self.weightsERI += self.stdpKernel(self.learningRate *\
                                           self.learnFactorEI *\
                                           self.dt *\
                                           self.activationsER, I, t,
                                           False, True)

      for I, EL, ER, i in self.activationBuffer:
        t = (time - i) * self.dt
        self.weightsII +=  self.stdpKernel(self.learningRate *\
                                           self.learnFactorII *\
                                           self.dt *\
                                           I, self.activationsI, t,
                                           True, True)

        self.weightsIEL += self.stdpKernel(self.learningRate *\
                                           self.learnFactorIE *\
                                           self.dt *\
                                           I, self.activationsEL, t,
                                           True, False)

        self.weightsIER += self.stdpKernel(self.learningRate *\
                                           self.learnFactorIE *\
                                           self.dt *\
                                           I, self.activationsER, t,
                                           True, False)

        self.weightsELI += self.stdpKernel(self.learningRate *\
                                           self.learnFactorEI *\
                                           self.dt *\
                                           EL, self.activationsI, t,
                                           False, True)

        self.weightsERI += self.stdpKernel(self.learningRate *\
                                           self.learnFactorEI *\
                                           self.dt *\
                                           ER, self.activationsI, t,
                                           False, True)

      self.activationBuffer.append((np.copy(self.activationsI),
                                    np.copy(self.activationsEL),
                                    np.copy(self.activationsER), time))
