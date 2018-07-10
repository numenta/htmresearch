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
import matplotlib.animation as animation
import copy
import os
from compute_hardwired_weights import compute_hardwired_weights

# STDP kernel time constant in seconds.  Used for the default kernel.
STDP_TIME_CONSTANT = 0.012

# How often to update plots
PLOT_INTERVAL = 0.1

# How often path integration estimates are collected.  This needs to be tuned,
# as values that are too small will lead to constant estimates of zero movement.
ESTIMATION_INTERVAL = 0.25

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
    # E-I, Hebbian
    stdpScaler *= 1
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

  timeFactor = np.exp(-1*np.abs(dt)/(STDP_TIME_CONSTANT*stdpTimeScaler))

  updates = np.outer(preSynActivation*timeFactor*np.sign(dt)*stdpScaler,
                     postSynActivation)

  return updates

def placeSTDPKernel(placeActivation, otherActivation, dt):
  timeFactor = np.exp(-1*np.abs(dt)/(STDP_TIME_CONSTANT))
  updates = np.outer(placeActivation*timeFactor*np.sign(dt), otherActivation)
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


class Dynamic1DCAN(object):
  def __init__(self,
               numExcitatory,
               numInhibitory,
               numPlaces,
               learningRate,
               dt,
               stdpWindow=10,
               decayConstant=0.03,
               velocityGain=0.9,
               placeGainE=10,
               placeGainI=50,
               sigmaLoc=0.01,
               stdpKernel=defaultSTDPKernel,
               tonicMagnitude=0,
               learnFactorII=7,
               learnFactorEI=2,
               learnFactorIE=1,
               learnFactorP=1,
               envelopeWidth=0.8,
               envelopeFactor=25,
               initialWeightScale=0.003,
               clip=10,
               plotting=True,
               hardwireI=True,
               boostEffect=10,
               boostDecay=0.01,
               maxWeightEI=10,
               maxWeightPI=1,
               maxWeightPE=3,
               boostGradient=1.,
               gradientType="linear",
               IEWeightFactor=1.,
               placeWeightInitialScale=.25,
               roomSize=1.,
               ):
    """

    :param numExcitatory: Size of each excitatory population.  Note that there
            are several populations, each of which has this many cells.
    :param numInhibitory: Size of inhibitory population.
    :param numPlaces: The size of the incoming place code. Not all must be used.
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
    :param tonicMagnitude: The magnitude of the global tonic input
            during training.
    :param learnFactorII: Extra learning rate for II connections.
    :param learnFactorEI: Extra learning rate for EI connections.
    :param learnFactorIE: Extra learning rate for IE connections.
    :param learnFactorP: Extra learning rate for P connections.
    :param envelopeWidth: The distance away from a boundary at which
             the suppressive envelope is first applied.
    :param envelopeFactor: The steepness of the suppressive envelope.
    :param initialWeightScale: The maximum initial weight value.
    :param clip: The maximum possible activation.
    :param plotting: Whether or not to generate plots.  False speeds training.
    :param hardwireI: Whether or not to hardwire the inhibitory weights.
    :param boostEffect: How strong a boosting effect to use to promote activity
            of underused neurons.  0 is disabled.  Only used in early learning.
    :param boostDecay: How fast boosting history decays
    :param boostGradient: How much stronger boosting should be on the right side
            of the sheet than the left side.
    :param gradientType: If enabled, sets boosting gradient to be linear or
            exponential.
    :param roomSize: The size of the initial learning room (length in meters).
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

    # Place code weights
    self.weightsPI = np.random.random_sample((numPlaces, numInhibitory))* \
                          placeWeightInitialScale
    self.weightsPEL = np.random.random_sample((numPlaces, numExcitatory))* \
                          placeWeightInitialScale
    self.weightsPER = np.random.random_sample((numPlaces, numExcitatory))* \
                          placeWeightInitialScale

    # Determine a starting place code, which will govern activation
    # during learning.  This code is ignored during testing.
    self.placeCode = np.arange(0, roomSize, float(roomSize)/numPlaces)
    self.roomSize = roomSize

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
    self.activationsP = np.zeros((numPlaces,), dtype="float32")

    self.instantaneousI = np.zeros((numInhibitory,))
    self.instantaneousER = np.zeros((numExcitatory,))
    self.instantaneousEL = np.zeros((numExcitatory,))

    self.activationHistoryI = np.zeros((numInhibitory,), dtype="float32")
    self.activationHistoryER = np.zeros((numExcitatory,), dtype="float32")
    self.activationHistoryEL = np.zeros((numExcitatory,), dtype="float32")

    self.stdpWindow = stdpWindow
    self.stdpKernel = stdpKernel

    self.activationBuffer = deque(maxlen=int(self.stdpWindow))

    self.tonicMagnitude = tonicMagnitude

    self.envelopeWidth = envelopeWidth
    self.envelopeFactor = envelopeFactor

    self.learnFactorII = learnFactorII
    self.learnFactorEI = learnFactorEI
    self.learnFactorIE = learnFactorIE
    self.learnFactorP = learnFactorP

    self.envelopeI = self.computeEnvelope(np.linspace(0, 1, numInhibitory))
    self.envelopeE = self.computeEnvelope(np.linspace(0, 1, numExcitatory))

    self.clip = clip
    self.maxWeightEI = maxWeightEI
    self.maxWeightPI = maxWeightPI
    self.maxWeightPE = maxWeightPE

    self.plotting = plotting

    if "linear" in gradientType.lower():
      self.boostEffectI = np.linspace(1, boostGradient, numInhibitory)*boostEffect
      self.boostEffectE = np.linspace(1, boostGradient, numExcitatory)*boostEffect
    elif "exponential" in gradientType.lower():
      base = np.power(boostGradient, 1./numInhibitory)
      self.boostEffectI = np.power(base, np.arange(0, numInhibitory))
      base = np.power(boostGradient, 1./numExcitatory)
      self.boostEffectE = np.power(base, np.arange(0, numExcitatory))

    self.alpha = boostDecay

    self.weightFilter = np.zeros((numExcitatory, numInhibitory), dtype="float32")
    for i in range(self.weightFilter.shape[0]):
      for j in range(self.weightFilter.shape[1]):
        offset = np.abs(float(i)/self.weightFilter.shape[0] -
                        float(j)/self.weightFilter.shape[1])
        if offset < 0.1:
          self.weightFilter[i, j] = 1.

    if hardwireI:
      self.hardwireWeights(False, True, True)
      self.weightsII *= self.weightFilter
      self.weightsIER *= self.weightFilter
      self.weightsIEL *= self.weightFilter
      self.weightsIEL *= IEWeightFactor
      self.weightsIER *= IEWeightFactor



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
                       for i in range(-10, 11, 1)]

          shift = np.argmin(rotations) - 10

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

          self.plotActivation(time=t, velocity=v, boosting=False)

    self.dt = oldDt
    return(np.asarray(trueVelocities), np.asarray(estimatedVelocities))


  def hardwireWeights(self, flip=False, onlyI=False, constantIE=False):
    (G_I_EL, G_I_ER, G_EL_I, G_ER_I, G_I_I) = \
      compute_hardwired_weights(2.2,
                                self.activationsEL.shape[0],
                                self.activationsI.shape[0],
                                True,
                                onlyI=onlyI)

    # We need to flip the signs for the inhibitory weights;
    # in our convention, inhibitory weights are always negative,
    # but in theirs, they are positive and the sign flip is applied
    # during activation.
    self.weightsII = -1.*G_I_I

    if constantIE:
      self.weightsIER = np.copy(-1.*G_I_I)
      self.weightsIEL = np.copy(-1.*G_I_I)
      return


    # If we want the network to path integrate in the right direction,
    # flip ELI and ERI.
    if not onliI and flip:
      self.weightsELI = G_ER_I
      self.weightsERI = G_EL_I
      self.weightsIEL = -1.*G_I_ER
      self.weightsIER = -1.*G_I_EL

    elif not onlyI:
      self.weightsELI = G_EL_I
      self.weightsERI = G_ER_I
      self.weightsIEL = -1. * G_I_EL
      self.weightsIER = -1. * G_I_ER


  def updatePlaceWeights(self):
    """
    We use a simplified version of Hebbian learning to learn place weights.
    Cells above the boost target are wired to the currently-active places,
    cells below it have their connection strength to them reduced.
    """
    self.weightsPI += np.outer(self.activationsI - self.boostTarget,
                               self.activationsP)*self.dt*\
                               self.learnFactorP*self.learningRate
    self.weightsPEL += np.outer(self.activationsEL - self.boostTarget,
                               self.activationsP)*self.dt*\
                               self.learnFactorP*self.learningRate
    self.weightsPER += np.outer(self.activationsER - self.boostTarget,
                               self.activationsP)*self.dt*\
                               self.learnFactorP*self.learningRate

    np.minimum(self.weightsPI, 1, self.weightsPI)
    np.minimum(self.weightsPEL, 1, self.weightsPEL)
    np.minimum(self.weightsPER, 1, self.weightsPER)


  def simulate(self, time,
               feedforwardInputI,
               feedforwardInputE,
               v,
               recurrent=True,
               dt=None,
               envelope=False,
               inputNoise=None,
               sampleFreq=1,
               startFrom=0,
               save=True):
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
      self.ax1 = self.fig.add_subplot(311)
      self.ax2 = self.fig.add_subplot(312)
      self.ax3 = self.fig.add_subplot(313)
      plt.ion()

      self.ax1.set_xlabel("Excitatory population activity")
      self.ax2.set_xlabel("Inhibitory population activity")

      plt.tight_layout()
      self.fig.show()
      self.fig.canvas.draw()

    self.activationsI = np.random.random_sample(self.activationsI.shape)
    self.activationsEL = np.random.random_sample(self.activationsEL.shape)
    self.activationsER = np.random.random_sample(self.activationsER.shape)
    self.activationsP.fill(0)
    self.activationHistoryI.fill(0)
    self.activationHistoryEL.fill(0)
    self.activationHistoryER.fill(0)

    if dt is None:
      oldDt = self.dt
    else:
      oldDt = self.dt
      self.dt = dt
    times = np.arange(0, time, self.dt)
    samples = np.arange(startFrom, time, self.dt)
    results = np.zeros((len(samples)/sampleFreq, len(self.activationsI)))
    s = 0
    for i, t in enumerate(times):
      if inputNoise is not None:
        noisesI = np.random.random_sample(feedforwardInputI.shape)*inputNoise
        noisesE = np.random.random_sample(feedforwardInputE.shape)*inputNoise
      else:
        noisesE = 1.; noisesI = 1.

      self.activationsP = np.zeros(self.activationsP.shape)

      self.update(feedforwardInputI*noisesI, feedforwardInputE*noisesE,
                  v, recurrent, envelope=envelope)


      if i % sampleFreq == 0 and t >= startFrom and save:
        results[s] = self.activationsI
        print("At {}".format(t))
        s += 1


      if self.plotting:
        plotTime = np.abs(np.mod(t, PLOT_INTERVAL))
        if plotTime <= 0.00001 or np.abs(plotTime - PLOT_INTERVAL) <= 0.00001:
          self.plotActivation(time=t, velocity=v)

    self.dt = oldDt

    if save:
      return results


  def update(self, feedforwardInputI, feedforwardInputE, v, recurrent=True,
             envelope=False, iSpeedTuning=False, enforceDale=True):
    """
    Do one update of the CAN network, of length self.dt.
    :param feedforwardInputI: The feedforward input to inhibitory cells.
    :param feedforwardInputR: The feedforward input to excitatory cells.
    :param placeActivity: Activity of the place code.
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

    np.matmul(self.activationsP * self.placeGainI, self.weightsPI,
              self.instantaneousI)
    np.matmul(self.activationsP* self.placeGainE, self.weightsPEL,
              self.instantaneousEL)
    np.matmul(self.activationsP * self.placeGainE, self.weightsPER,
              self.instantaneousER)

    self.instantaneousI += self.boostEffectI*\
                           self.activationHistoryI +\
                           feedforwardInputI
    self.instantaneousEL += self.boostEffectE*\
                            self.activationHistoryEL +\
                            feedforwardInputE
    self.instantaneousER += self.boostEffectE*\
                            self.activationHistoryER +\
                            feedforwardInputE

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
      self.instantaneousI += (np.matmul(self.activationsEL, weightsELI) +\
                np.matmul(self.activationsER, weightsERI) +\
                np.matmul(self.activationsI, weightsII))

      self.instantaneousEL += np.matmul(self.activationsI, weightsIEL)
      self.instantaneousER += np.matmul(self.activationsI, weightsIER)

    self.instantaneousI += self.tonicMagnitude
    self.instantaneousEL += self.tonicMagnitude
    self.instantaneousER += self.tonicMagnitude

    self.instantaneousEL *= max((1 - self.velocityGain*v), 0)
    self.instantaneousER *= max((1 + self.velocityGain*v), 0)
    if iSpeedTuning:
      self.instantaneousI *= min(self.velocityGain*np.abs(v), 1)


    if envelope:
      self.instantaneousI *= self.envelopeI
      self.instantaneousER *= self.envelopeE
      self.instantaneousEL *= self.envelopeE

    # Input must be positive.
    np.maximum(self.instantaneousI, 0., self.instantaneousI)
    np.maximum(self.instantaneousEL, 0., self.instantaneousEL)
    np.maximum(self.instantaneousER, 0., self.instantaneousER)

    # Activity decay and timestep adjustment
    self.activationsI += (self.instantaneousI - self.activationsI/self.decayConstant)*self.dt
    self.activationsEL += (self.instantaneousEL - self.activationsEL/self.decayConstant)*self.dt
    self.activationsER += (self.instantaneousER - self.activationsER/self.decayConstant)*self.dt

    # Finally, clip activations for stability
    np.minimum(self.activationsI, self.clip, self.activationsI)
    np.minimum(self.activationsEL, self.clip, self.activationsEL)
    np.minimum(self.activationsER, self.clip, self.activationsER)

    self.activationHistoryI += (-self.activationsI + np.sum(self.activationsI)/np.sum(self.envelopeI))*self.dt
    self.activationHistoryEL += (-self.activationsEL + np.sum(self.activationsEL)/np.sum(self.envelopeE))*self.dt
    self.activationHistoryER += (-self.activationsER + np.sum(self.activationsER)/np.sum(self.envelopeE))*self.dt

    self.activationHistoryI -= self.dt*self.activationHistoryI/self.alpha
    self.activationHistoryEL -= self.dt*self.activationHistoryEL/self.alpha
    self.activationHistoryER -= self.dt*self.activationHistoryER/self.alpha

    # self.activationHistoryI +=  self.activationHistoryI * (self.alpha) + \
    #                             (1 - self.alpha)*\
    #                             (-self.activationsI + np.sum(self.activationsI)/np.sum(self.envelopeI))*self.dt
    # self.activationHistoryEL += self.activationHistoryEL * (self.alpha) + \
    #                             (1 - self.alpha)*\
    #                             (-self.activationsEL + np.sum(self.activationsEL)/np.sum(self.envelopeE))*self.dt
    # self.activationHistoryER += self.activationHistoryER * (self.alpha) + \
    #                             (1 - self.alpha)*\
    #                             (-self.activationsER + np.sum(self.activationsER)/np.sum(self.envelopeE))*self.dt


    #
    # self.activationHistoryI * (np.max(np.abs(v) * self.alpha, 0)) + (1 - np.max(np.abs(v) * self.alpha, 0))
    # self.activationHistoryEL * (np.max(np.abs(v) * self.alpha, 0)) + (1 - np.max(np.abs(v) * self.alpha, 0))
    # self.activationHistoryER * (np.max(np.abs(v) * self.alpha, 0)) + (1 - np.max(np.abs(v) * self.alpha, 0))

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
             randomSpeed=False,
             learnRecurrent=False,
             envelope=True,):
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
    :param learnRecurrent: Whether or not to learn recurrent connections.
    :param envelope: Whether or not the envelope should be active in learning.
    """

    # Simulate for a second to get nice starting activation bumps.
    # Turn plotting off so as not to confuse the viewer
    oldPlotting = self.plotting
    self.plotting = False
    self.simulate(10, 1, 1, 0, envelope=False, inputNoise=None, save=False)
    self.plotting = oldPlotting

    # Set up plotting
    if self.plotting:
      self.fig = plt.figure()
      self.ax1 = self.fig.add_subplot(611)
      self.ax2 = self.fig.add_subplot(612)
      self.ax3 = self.fig.add_subplot(613)
      self.ax4 = self.fig.add_subplot(212)

      self.ax3.set_xlabel("Inhibitory-Inhibitory connections")
      plt.ion()
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
        length = 10. / speed

        runTimes = np.arange(0, length, self.dt)
        trajectory = (np.sin(dir * (runTimes * np.pi / (5 / speed) - np.pi / 2.)) + 1) * \
                     2.5 + startingPoint
        trajectories.append(trajectory)
        timings.append(runTimes + time)
        time += length
        startingPoint += 1. / runs

    for trajectory, timing in zip(trajectories, timings):
      self.activationsI = np.random.random_sample(self.activationsI.shape)
      self.activationsER = np.random.random_sample(self.activationsER.shape)
      self.activationsEL = np.random.random_sample(self.activationsEL.shape)
      self.activationHistoryI = np.zeros(self.activationsI.shape)
      self.activationHistoryEL = np.zeros(self.activationsEL.shape)
      self.activationHistoryER = np.zeros(self.activationsER.shape)
      velocities = np.diff(trajectory)/self.dt
      for i, t in enumerate(timing[:-1]):
        x = trajectory[i] % self.roomSize
        v = velocities[i]
        self.activationsP = np.exp(-1.*(self.placeCode - x)**2 /
                              (2*self.sigmaLoc**2))

        self.update(0, 0, v,
                    recurrent=recurrent,
                    envelope=envelope,
                    iSpeedTuning=periodic,
                    enforceDale=True,
                    )

        self.stdpUpdate(t, onlyPlace=not learnRecurrent)

        # Enforce Dale's law for place cells.  Place cells must be excitatory.
        np.maximum(self.weightsPI, 0, self.weightsPI)
        np.maximum(self.weightsPEL, 0, self.weightsPEL)
        np.maximum(self.weightsPER, 0, self.weightsPER)

        # Also keep the place weights from being too large.
        np.minimum(self.weightsPI, 1., self.weightsPI)
        np.minimum(self.weightsPEL, 3., self.weightsPEL)
        np.minimum(self.weightsPER, 3., self.weightsPER)

        if self.plotting:
          residTime += self.dt
          if residTime > PLOT_INTERVAL:
            residTime -= PLOT_INTERVAL
            self.ax4.matshow(self.weightsPI, cmap=plt.cm.coolwarm)
            self.plotActivation(position=x, time=t)
      self.stdpUpdate(t, onlyPlace=not learnRecurrent, clearBuffer=True)

    # Finally, enforce Dale's law for recurrent connections.
    # Inhibitory neurons must be inhibitory,
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


  def plotActivation(self, position=None, time=None, velocity=None, boosting=True):
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
    self.ax1.legend(loc = "best")

    self.ax2.clear()
    x = np.arange(0, len(self.activationsI), 1)
    self.ax2.plot(x, self.activationsI, color = "k", label = "I Activation")
    self.ax2.legend(loc = "best")

    if boosting:
      self.ax3.clear()
      x = np.arange(0, len(self.activationHistoryI), 1)
      self.ax3.plot(x, self.activationHistoryI*self.boostEffectI, color = "k", label = "Boost history")
      if position is not None:
        self.ax3.axvline(x=position*len(self.activationsI))
      self.ax3.legend(loc = "best")
      self.ax3.set_xlabel("Boosting")

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

  def stdpUpdate(self, time, clearBuffer=False, onlyPlace=False):
    """
    Adds the current activations to the tracking queue, and then performs an
    STDP update if possible.
    :param time: The current time.  Must be provided.
    :param clearBuffer: Set as True to clear the activation buffer.
            This should be done at the end of training.
    :param onlyPlace: Only learn place connections.
    """
    if clearBuffer:
      while len(self.activationBuffer) > 1:
        baseI, baseEL, baseER, baseP, t = self.activationBuffer.popleft()
        for (I, EL, ER, P, i) in self.activationBuffer:
          t = (i - t) * self.dt
          self.weightsPI += placeSTDPKernel(self.learningRate *\
                                            self.learnFactorP *\
                                            self.dt *\
                                            baseP, I, t,
                                            )

          self.weightsPEL += placeSTDPKernel(self.learningRate *\
                                             self.learnFactorP *\
                                             self.dt *\
                                             baseP, EL, t,
                                             )

          self.weightsPER += placeSTDPKernel(self.learningRate *\
                                             self.learnFactorP *\
                                             self.dt *\
                                             baseP, ER, t,
                                             )

          if not onlyPlace:
            self.weightsELI += self.stdpKernel(self.learningRate *\
                                               self.learnFactorEI *\
                                               self.dt *\
                                               baseEL, I, t,
                                               False, True)

            self.weightsERI += self.stdpKernel(self.learningRate *\
                                               self.learnFactorEI *\
                                               self.dt *\
                                               baseER, I, t,
                                               False, True)

    else:
      for I, EL, ER, P, i in reversed(self.activationBuffer):
        t = (i - time) * self.dt
        self.weightsPI +=  placeSTDPKernel(self.learningRate *\
                                           self.learnFactorP *\
                                           self.dt *\
                                           self.activationsP, I, t,
                                           )

        self.weightsPEL += placeSTDPKernel(self.learningRate *\
                                           self.learnFactorP *\
                                           self.dt *\
                                           self.activationsP, EL, t,
                                           )

        self.weightsPER += placeSTDPKernel(self.learningRate *\
                                           self.learnFactorP *\
                                           self.dt *\
                                           self.activationsP, ER, t,
                                           )

        if not onlyPlace:
          self.weightsELI += self.stdpKernel(self.learningRate *\
                                             self.learnFactorEI *\
                                             self.dt *\
                                             self.instantaneousEL, I, t,
                                             False, True)

          self.weightsERI += self.stdpKernel(self.learningRate *\
                                             self.learnFactorEI *\
                                             self.dt *\
                                             self.instantaneousER, I, t,
                                             False, True)

      for I, EL, ER, P, i in self.activationBuffer:
        t = (time - i) * self.dt
        self.weightsPI +=  placeSTDPKernel(self.learningRate *\
                                           self.learnFactorP *\
                                           self.dt *\
                                           P, self.instantaneousI, t,
                                           )

        self.weightsPEL += placeSTDPKernel(self.learningRate *\
                                           self.learnFactorP *\
                                           self.dt *\
                                           P, self.instantaneousEL, t,
                                           )

        self.weightsPER += placeSTDPKernel(self.learningRate *\
                                           self.learnFactorP *\
                                           self.dt *\
                                           P, self.instantaneousER, t,
                                           )

        if not onlyPlace:
          self.weightsELI += self.stdpKernel(self.learningRate *\
                                             self.learnFactorEI *\
                                             self.dt *\
                                             EL, self.instantaneousI, t,
                                             False, True)

          self.weightsERI += self.stdpKernel(self.learningRate *\
                                             self.learnFactorEI *\
                                             self.dt *\
                                             ER, self.instantaneousI, t,
                                             False, True)

      self.activationBuffer.append((np.copy(self.instantaneousI),
                                    np.copy(self.instantaneousEL),
                                    np.copy(self.instantaneousER),
                                    np.copy(self.activationsP),
                                    time))

def w_0(x):
  """
  @param x (numpy array)
  A point
  """
  a = 1.0
  lambda_net = 13.0
  beta = 3.0 / lambda_net ** 2
  gamma = 1.05 * beta

  x_length_squared = x[0] ** 2 + x[1] ** 2
  return a * np.exp(-gamma * x_length_squared) - np.exp(-beta * x_length_squared)


class Dynamic2DCAN(Dynamic1DCAN):
  def __init__(self,
               numPlaces,
               learningRate,
               dt,
               dimensions=(32, 32),
               stdpWindow=10,
               decayConstant=0.03,
               velocityGain=0.9,
               placeGainE=10,
               placeGainI=50,
               sigmaLoc=0.01,
               stdpKernel=defaultSTDPKernel,
               tonicMagnitude=0,
               learnFactorII=7,
               learnFactorEI=2,
               learnFactorIE=1,
               learnFactorP=1,
               envelopeWidth=0.8,
               envelopeFactor=25,
               initialWeightScale=0.003,
               clip=10,
               plotting=True,
               movie=True,
               hardwireI=True,
               boostEffect=10,
               boostTarget=0.1,
               periodic=True,
               hardwireEnvelope=False,):
    """
    :param dimensions: The number of neuron groups in each direction
            on the sheet.  2-tuple.  Will have the product as number of groups.
    :param numPlaces: The size of the incoming place code. Not all must be used.
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
    :param tonicMagnitude: The magnitude of the global tonic input
            during training.
    :param learnFactorII: Extra learning rate for II connections.
    :param learnFactorEI: Extra learning rate for EI connections.
    :param learnFactorIE: Extra learning rate for IE connections.
    :param learnFactorP: Extra learning rate for P connections.
    :param envelopeWidth: The distance away from a boundary at which
             the suppressive envelope is first applied.
    :param envelopeFactor: The steepness of the suppressive envelope.
    :param initialWeightScale: The maximum initial weight value.
    :param clip: The maximum possible activation.
    :param plotting: Whether or not to generate plots.  False speeds training.
    :param movie: Whether or not to generate movies of simulations and training.
    :param hardwireI: Whether or not to hardwire the inhibitory weights.
    :param boostEffect: How strong a boosting effect to use to promote activity
            of underused neurons.  0 is disabled.  Only used in early learning.
    :param boostDecay: The decay rate of boosting history
    :param periodic: Whether or not to use toroidal weight structures.
    :param hardwireEnvelope: Whether or not to weaken connections to enveloped
            cells instead of suppressing them.

    """

    # Determine a starting place code, which will govern activation
    # during learning.  This code is ignored during testing.
    self.numPlaces = numPlaces
    self.placeCode = np.random.random((self.numPlaces, 2))


    self.placeGainE = placeGainE
    self.placeGainI = placeGainI
    self.velocityGain = velocityGain

    self.sigmaLoc = sigmaLoc

    self.learningRate = learningRate
    self.dt = dt
    self.decayConstant = decayConstant

    self.numInhibitory = dimensions[0]*dimensions[1]
    self.numExcitatory = dimensions[0]*dimensions[1]
    self.dimensions = dimensions

    self.activationsI = np.zeros((self.numInhibitory,))
    self.instantaneousI = np.zeros((self.numInhibitory,))
    self.activationHistoryI = np.zeros((self.numInhibitory,))
    self.activationsP = np.zeros((numPlaces,))

    self.envelopeWidth = envelopeWidth
    self.envelopeFactor = envelopeFactor

    self.directions = {"n": np.array([-1.0, 0.0]),
                       "e": np.array([0.0, 1.0]),
                       "s": np.array([1.0, 0.0]),
                       "w": np.array([0.0, -1.0])
                       }

    self.activations = dict((k, np.zeros(self.numExcitatory))
                            for k in self.directions.iterkeys())

    self.instantaneous = dict((k, np.zeros(self.numExcitatory,))
                            for k in self.directions.iterkeys())

    self.histories = dict((k, np.zeros(self.numExcitatory))
                            for k in self.directions.iterkeys())

    self.weightsEI = dict((k, np.zeros((self.numInhibitory, self.numExcitatory)))
                                 for k in self.directions.iterkeys())
    self.weightsIE = dict((k, np.zeros((self.numExcitatory, self.numInhibitory)))
                                 for k in self.directions.iterkeys())
    self.weightsII = np.zeros((self.numInhibitory, self.numInhibitory))
    self.weightsPE = dict((k, np.zeros((self.numExcitatory, self.numPlaces)))
                                 for k in self.directions.iterkeys())
    self.weightsPI = np.zeros((self.numInhibitory, self.numPlaces))

    self.stdpWindow = stdpWindow
    self.stdpKernel = stdpKernel

    self.activationBuffer = deque(maxlen=int(self.stdpWindow))

    self.tonicMagnitude = tonicMagnitude

    self.learnFactorII = learnFactorII
    self.learnFactorEI = learnFactorEI
    self.learnFactorIE = learnFactorIE
    self.learnFactorP = learnFactorP

    self.clip = clip
    self.plotting = plotting

    self.boostEffect = boostEffect
    self.boostTarget = boostTarget

    self.envelope = self.computeEnvelope()

    if hardwireI:
      # Calculate it once
      if periodic:
        jCoord0 = np.unravel_index(0, self.dimensions)

        jTargetCoord = np.mod(jCoord0, self.dimensions[0])

        weights = np.zeros(self.dimensions, dtype="float")

        for i in xrange(self.numInhibitory):
          iCoord = np.unravel_index(i, self.dimensions)


          distanceComponents1 = np.abs(iCoord - jTargetCoord)

          # The two points might actually be closer by wrapping around one/two of the edges.
          # For each dimension, consider what the alternate distance would have been,
          # and choose the lowest.
          distanceComponents2 = float(self.dimensions[0]) - distanceComponents1
          distanceComponents = np.where(distanceComponents1 < distanceComponents2,
                                        distanceComponents1, distanceComponents2)

        weights[iCoord, jCoord0] = 1000*w_0(distanceComponents*2)

        for j in xrange(self.numInhibitory):
          jCoord = np.unravel_index(j, self.dimensions)

          for k, preferredDirection in self.directions.iteritems():
            self.weightsIE[k][:, j] = np.roll(np.roll(weights, jCoord[0], axis=0),
                                              jCoord[1], axis=1).flatten()

          self.weightsII[:, j] = np.roll(np.roll(weights, jCoord[0], axis=0),
                                         jCoord[1], axis=1).flatten()

      else:
        if os.path.isfile(str(self.dimensions)+"weightCache.npz"):
          weights = np.load(str(self.dimensions)+"weightCache.npz")
          self.weightsII = np.copy(weights)
          for k, w in self.weightsIE.items():
            w = np.copy(weights)
        else:
          for j in range(self.numInhibitory):
            jCoord = np.unravel_index(j, self.dimensions)

            for i in xrange(self.numInhibitory):
              iCoord = np.unravel_index(i, self.dimensions)
              distanceComponents = np.abs(np.asarray(iCoord) - np.asarray(jCoord))
              if distanceComponents[0] + distanceComponents[1] > 20:
                continue

              weights = 1000*w_0(distanceComponents*2)

              for k, preferredDirection in self.directions.iteritems():
                self.weightsIE[k][i, j] = weights

              self.weightsII[i, j] = weights

        np.save(str(self.dimensions)+"weightCache.npz", self.weightsII)

    if hardwireEnvelope:
      self.weightsII *= self.envelope
      for k, w in self.weightsIE.items():
        w *= self.envelope

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
    """
    # Set up plotting
    if self.plotting:
      self.fig = plt.figure()
      self.ax1 = self.fig.add_subplot(221)
      self.ax2 = self.fig.add_subplot(222)
      self.ax3 = self.fig.add_subplot(223)
      plt.ion()

      self.ax1.set_xlabel("Excitatory population activity")
      self.ax2.set_xlabel("Inhibitory population activity")

      plt.tight_layout()
      self.fig.show()
      self.fig.canvas.draw()

    if self.movie:
      history = []

    self.activationsI = np.random.random_sample(self.activationsI.shape)
    for k, a in self.activations.items():
      self.a = np.random.random_sample(a.shape)

    self.activationsP.fill(0)
    self.activationHistoryI.fill(0)
    residTime = 0.
    for k, a in self.histories.items():
      a.fill(0)

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

      self.activationsP = np.zeros(self.activationsP.shape)

      self.update(feedforwardInputI*noisesI, feedforwardInputE*noisesE,
                  v, recurrent, envelope=envelope)

      residTime += self.dt
      if residTime > PLOT_INTERVAL:
        residTime -= PLOT_INTERVAL
        if self.plotting:
          self.plotActivation(time=t, velocity=v)
        if self.movie:
          history.append(np.copy(self.activationsI))
          print(t)

    if self.movie:
      self.createMovie(np.stack(history, -1), "IMovement2D.mp4",
                       self.numInhibitory, self.numPlaces)

    self.dt = oldDt

  def update(self, feedforwardInputI, feedforwardInputE, v, recurrent=True,
             envelope=False, iSpeedTuning=False):
    """
    Do one update of the CAN network, of length self.dt.
    :param feedforwardInputI: The feedforward input to inhibitory cells.
    :param feedforwardInputR: The feedforward input to excitatory cells.
    :param placeActivity: Activity of the place code.
    :param v: The current velocity.
    :param recurrent: Whether or not recurrent connections should be used.
    :param envelope: Whether or not an envelope should be applied.
    :param iSpeedTuning: Whether or not inhibitory cells should also have their
             activations partially depend on current movement speed.  This is
             necessary for periodic training, serving a role similar to that of
             the envelope.
    """
    np.matmul(self.weightsPI, self.activationsP*self.placeGainI,
              self.instantaneousI)
    if recurrent:
      self.instantaneousI += np.matmul(np.minimum(self.weightsII, 0),
                                       self.activationsI,
                                       )
    self.instantaneousI += self.boostEffect*self.activationHistoryI +\
                           feedforwardInputI + self.tonicMagnitude

    if envelope:
      self.instantaneousI *= self.envelope

    if iSpeedTuning:
      self.instantaneousI *= min(self.velocityGain * np.abs(v), 1)

    np.maximum(self.instantaneousI, 0., self.instantaneousI)

    for k, a in self.instantaneous.items():
      #np.matmul(self.weightsPE[k], self.activationsP * self.placeGainE,  a)
      #a += feedforwardInputE + self.tonicMagnitude +\
      #     self.boostEffect*self.histories[k]
      #if recurrent:
      #  a += np.matmul(np.minimum(self.weightsIE[k], 0), self.activationsI)
      #  self.instantaneousI += np.matmul(np.maximum(self.weightsEI[k], 0),
      #                                   self.activations[k]
       #                                  )

      a *= max((1 + np.dot(v, self.directions[k]), 0))

      # Input must be positive
      np.maximum(a, 0, a)

      if envelope:
        a *= self.envelope

      # Activity decay and timestep adjustment
      self.activations[k] += (a - self.activations[k]/self.decayConstant)*self.dt

      # Finally, clip activations for stability
      np.minimum(self.activations[k], self.clip, self.activations[k])

      self.histories[k] -= (self.activations[k]
                                    - np.mean(self.activations[k]))*self.dt

    # Activity decay and timestep adjustment
    self.activationsI += (self.instantaneousI -
                          self.activationsI/self.decayConstant)*self.dt

    # Finally, clip activations for stability
    np.minimum(self.activationsI, self.clip, self.activationsI)

    self.activationHistoryI -= (self.activationsI -
                                np.mean(self.activationsI))*self.dt


  def computeEnvelope(self):
    """
    Compute an envelope for use in suppressing border cells.
    :return: A numpy array that can be elementwise-multiplied with activations
             for the given cell population to apply the envelope.
    """
    envelopeX = [1 if np.abs(p) < 1 - self.envelopeWidth else
                      np.exp(-1.*self.envelopeFactor *
                      ((np.abs(p) - 1 + self.envelopeWidth)/self.envelopeWidth)**2)
                      for p in np.linspace(-1, 1, self.dimensions[0])]

    envelopeY = [1 if np.abs(p) < 1 - self.envelopeWidth else
                      np.exp(-1.*self.envelopeFactor *
                      ((np.abs(p) - 1 + self.envelopeWidth)/self.envelopeWidth)**2)
                      for p in np.linspace(-1, 1, self.dimensions[1])]

    return np.asarray(np.outer(envelopeX, envelopeY).flatten())


  def decayWeights(self, decayConst=60):
    """
    Decay the network's weights.

    :param decayConst: The time constant (in seconds) to use for decay.
            Note: If applied, decay must be used extremely carefully, as
            it has a tendency to cause asymmetries in the network weights.
    """
    self.weightsII -= self.weightsII*self.dt/decayConst
    self.weightsPI -= self.weightsPI*self.dt/decayConst
    for k, w in self.weightsIE.items():
      w -= w*self.dt/decayConst
    for k, w in self.weightsEI.items():
      w -= w*self.dt/decayConst
    for k, w in self.weightsPE.items():
      w -= w*self.dt/decayConst


  def learnPlaceCode(self,
                     runs,
                     dir=1,
                     periodic=False,
                     recurrent=True,
                     randomSpeed=False,
                     learnRecurrent=False):
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
    :param learnRecurrent: Whether or not to learn recurrent connections.
    """

    # Simulate for a second to get nice starting activation bumps.
    # Turn plotting off so as not to confuse the viewer
    self.plotting = False
    self.simulate(10, 1, 1, 0, envelope=False, inputNoise=None)
    self.plotting = True

    # Set up plotting
    if self.plotting:
      self.fig = plt.figure()
      self.ax1 = self.fig.add_subplot(411)
      self.ax2 = self.fig.add_subplot(412)
      self.ax3 = self.fig.add_subplot(413)
      self.ax4 = self.fig.add_subplot(414)

      plt.ion()
      plt.tight_layout()
      self.ax3.set_xlabel("Inhibitory-Inhibitory connections")
      self.fig.show()
      self.fig.canvas.draw()

    if self.movie:
      history = []


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
        length = 10. / speed

        runTimes = np.arange(0, length, self.dt)
        trajectory[:, 0] = (np.sin(dir * (runTimes * np.pi / (5 / speed) - np.pi / 2.)) + 1) * \
                           2.5 + startingPoint
        trajectory[:, 1] = (np.sin(dir * (runTimes * np.pi / (5 / speed) - np.pi / 2.)) + 1) * \
                           2.5
        trajectories.append(trajectory)
        timings.append(runTimes + time)
        time += length
        startingPoint += 1. / runs

    for trajectory, timing in zip(trajectories, timings):
      self.activationsI = np.zeros(self.activationsI.shape)
      self.activationsER = np.zeros(self.activationsER.shape)
      self.activationsEL = np.zeros(self.activationsEL.shape)
      velocities = np.diff(trajectory)/self.dt

      for i, t in enumerate(timing[:-1]):
        x = trajectory[i] % 1
        v = velocities[i]
        self.activationsP = np.exp(-1.*(self.placeCode - x)**2 /
                              (2*self.sigmaLoc**2))

        self.update(0, 0, v,
                    recurrent=recurrent,
                    envelope=(not periodic),
                    iSpeedTuning=periodic,
                    enforceDale=True,
                    )

        self.stdpUpdate(t, onlyPlace=not learnRecurrent)

        # Finally, enforce Dale's law.  Place neurons must be excitatory.
        # Also keep the place weights from being too large.
        np.maximum(self.weightsPI, 0, self.weightsPI)
        np.minimum(self.weightsPI, 3., self.weightsPI)
        for k, w in self.weightsPE.items():
          np.maximum(w, 0, w)
          np.minimum(w, 3., w)

        residTime += self.dt
        if residTime > PLOT_INTERVAL:
          residTime -= PLOT_INTERVAL
          if self.plotting:
              self.ax4.matshow(self.weightsPI, cmap=plt.cm.coolwarm)
              self.plotActivation(position=x, time=t)
          if self.movie:
            history.append(np.copy(self.weightsPI))


      if self.movie:
        self.createMovie(np.stack(history, -1), "PIWeightEvolution",
                         self.numInhibitory, self.numPlaces)


      self.stdpUpdate(t, onlyPlace=not learnRecurrent, clearBuffer=True)

      # Enforce Dale's law
      np.minimum(self.weightsII, 0, self.weightsII)
      np.maximum(self.weightsPI, 0, self.weightsPI)
      for k, w in self.weightsIE.items():
        np.minimum(w, 0, w)
      for k, w in self.weightsEI.items():
        np.maximum(w, 0, w)
      for k, w in self.weightsPE.items():
        np.maximum(w, 0, w)


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
    y = self.activations["n"] + self.activations["s"] + self.activations["e"] + \
        self.activations["w"]
    self.ax1.matshow(y.reshape(self.dimensions))

    self.ax2.clear()
    self.ax2.matshow(self.activationsI.reshape(self.dimensions))

    self.ax3.clear()
    self.ax3.matshow(self.activationHistoryI.reshape(self.dimensions))

    titleString = ""
    if time is not None:
      titleString += "Time = {}".format(str(time))
    if velocity is not None:
      titleString += "  Velocity = {}".format(str(velocity)[:4])
    if position is not None:
      titleString += "  Position = {}".format(str(position)[:4])
    plt.suptitle(titleString)

    self.ax1.set_xlabel("Excitatory activity")
    self.ax2.set_xlabel("Inhibitory activity")
    self.ax3.set_xlabel("Boosting activity")

    plt.tight_layout()

    self.fig.canvas.draw()


  def stdpKernels(self, t, I, E, baseI, baseE, baseP, onlyPlace=False):
    self.weightsPI += placeSTDPKernel(self.learningRate * \
                                      self.learnFactorP * \
                                      self.dt * \
                                      baseP, I, t,
                                      )
    for k, a in E.items():
      self.weightsPE[k] += placeSTDPKernel(self.learningRate * \
                                           self.learnFactorP * \
                                           self.dt * \
                                           baseP, E[k], t,
                                           )

      if not onlyPlace:
        self.weightsEI[k] += self.stdpKernel(self.learningRate * \
                                             self.learnFactorP * \
                                             self.dt * \
                                             baseE[k], I, t,
                                             )
        self.weightsIE[k] += self.stdpKernel(self.learningRate * \
                                             self.learnFactorP * \
                                             self.dt * \
                                             baseI, E[k], t,
                                             )

  def createMovie(self, data, name, nx, ny):
    def update_line(num, data, line):
      line.set_data(data[num].reshape((nx, ny)))
      return line,

    fig = plt.figure()

    l = plt.imshow(data[0].reshape((nx, ny)), animated=True)
    ani = animation.FuncAnimation(fig, update_line, len(data), fargs=(data, l),
                                  interval=10, blit=True)

    ani.save('name.mp4')


  def stdpUpdate(self, time, clearBuffer=False, onlyPlace=False):
    """
    Adds the current activations to the tracking queue, and then performs an
    STDP update if possible.
    :param time: The current time.  Must be provided.
    :param clearBuffer: Set as True to clear the activation buffer.
            This should be done at the end of training.
    :param onlyPlace: Only learn place connections.
    """
    if clearBuffer:
      while len(self.activationBuffer) > 1:
        baseI, baseE, baseP, t = self.activationBuffer.popleft()
        for (I, E, P, i) in self.activationBuffer:
          t = (i - t) * self.dt
          self.sdtpKernels(t, I, E, baseI, baseE, baseP, onlyPlace=onlyPlace)

    else:
      for I, E, P, i in reversed(self.activationBuffer):
        t = (i - time) * self.dt
        self.sdtpKernels(t,
                         I,
                         E,
                         self.instantaneousI,
                         self.instantaneous,
                         self.activationsP,
                         onlyPlace=onlyPlace)

      for I, E, P, i in self.activationBuffer:
        t = (time - i) * self.dt
        self.sdtpKernels(t,
                         self.instantaneousI,
                         self.instantaneous,
                         I,
                         E,
                         P,
                         onlyPlace=onlyPlace)

      self.activationBuffer.append((np.copy(self.instantaneousI),
                                    {k: np.copy(self.instantaneous[k])
                                    for k in self.instantaneous},
                                    np.copy(self.activationsP),
                                    time))
