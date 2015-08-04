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

import numpy as np

from copy import copy


def divide(a, b):
  if a == 0:
    return 0
  return a/b


class HMM(object):
    def __init__(self, numCats, numStates, criterion=0.0001, verbosity=0):
      self.A = None # {a_ij} = P(X_t = j | X_t-1 = i)
      self.B = None # {b_ij} = P(Y_t = i | X_t = j)
      self.pi = None # {pi_i} = P(X_0 = i)
      self.numStates = numStates
      self.numCats = numCats
      self.observations = []
      self.verbosity = verbosity
      self.criterion = criterion

    def reset(self):
      self.observations = []

    def _initializeTrial(self, observations):
      self.observations = observations
      self.T = len(observations)
      self.seenValues = set(self.observations)
      self.alpha = np.zeros((self.numStates,self.T), dtype="float") # {a_it} = P(Y_1 = y_1, ..., Y_t=y_t, X_t=i | theta)
      self.beta = np.zeros((self.numStates,self.T), dtype="float") # {b_it} = P(Y_t+1 = y_t+1, ..., Y_T=y_T | X_t=i, theta)
      self.gamma = np.zeros((self.numStates,self.T), dtype="float") # {g_it} = P(X_t = i | Y, theta)
      self.eps = np.zeros((self.numStates,self.numStates,self.T), dtype="float") # {eps_ijt} = P(X_t = i, Xt+1 = j | Y, theta)

      if self.verbosity > 0:
        print "observations: ", observations

    def _forward(self):
      y1 = self.observations[0]

      for i in range(self.numStates):
        self.alpha[i,0] = self.pi[i]
        self.alpha[i,0] *= self.B[i,y1]


      for t in range(1, self.T):
        yt = self.observations[t]

        for j in range(0, self.numStates):
          sumAlphaT1 = 0.0

          for i in range(0, self.numStates):
            sumAlphaT1 += self.alpha[i,t-1]*self.A[i,j]

          self.alpha[j,t] = self.B[j, yt]*sumAlphaT1

      if self.verbosity > 0:
        print "alpha: ", self.alpha


    def _backward(self):
      for i in range(self.numStates):
        self.beta[i,self.T-1] = 1.0

      for t in range(self.T-1, 0, -1):
        yt = self.observations[t]

        for i in range(self.numStates):
          newBetaiT1 = 0.0

          for j in range(0, self.numStates):
            newBetaiT1 += self.beta[j,t]*self.A[i,j]*self.B[j,yt]

          self.beta[i,t-1] = newBetaiT1

      if self.verbosity > 0:
        print "beta: ", self.beta

    def _update(self):
      # updating gamma
      for t in range(self.T):
        denom = 0.0
        for i in range(self.numStates):
          denom += self.alpha[i,t]*self.beta[i,t]

        for i in range(self.numStates):
          self.gamma[i,t] = divide(self.alpha[i,t]*self.beta[i,t], denom)

      # updating eps
      for t in range(self.T-1):
        for i in range(self.numStates):
          denom = sum([self.alpha[j,t]*self.beta[j,t] for j in range(self.numStates)])
          yt1 = self.observations[t+1]
          for j in range(self.numStates):
            self.eps[i,j,t] = divide(self.alpha[i,t]*self.A[i,j]*self.beta[j,t+1]*self.B[j,yt1], denom)

      if self.verbosity > 0:
        print "gamma: ", self.gamma
        print "eps: ", self.eps

      # updating A
      denoms = np.zeros(self.numStates, dtype="float")
      for i in range(self.numStates):
        self.pi[i] = self.gamma[i, 0]
        for t in range(self.T-1):
          denoms[i] += self.gamma[i,t]

      for i in range(self.numStates):
        for j in range(self.numStates):
          numer = 0.0
          for t in range(self.T-1):
            numer += self.eps[i,j,t]

          self.A[i,j] = divide(numer, denoms[i])

      if self.verbosity > 0:
        print "A: ", self.A


      # updating B
      for i in range(self.numStates):
        for v in self.seenValues:
          numer = 0.0
          denom = 0.0
          for t in range(self.T):
            denom += self.gamma[i,t]
            if self.observations[t] == v:
              numer += self.gamma[i,t]
          self.B[i,v] = divide(numer, denom)

      if self.verbosity > 0:
        print "B: ", self.B


    def train(self, observations):
      self._initializeTrial(observations)

      while True:
        startA = copy(self.A)
        startB = copy(self.B)
        startpi = copy(self.pi)

        self._forward()
        self._backward()
        self._update()

        done = True

        if np.max(abs(startpi - self.pi)) > self.criterion:
          done = False
        elif np.max(abs(startA - self.A)) > self.criterion:
          done = False
        elif np.max(abs(startB - self.B)) > self.criterion:
          done = False

        if done:
          break


    def predict_next_inputs(self, current_input, threshold=0.3):
      next_inputs = set()

      self._initializeTrial([x for x in self.observations] + [current_input])
      t = len(self.observations)-1

      # P(X_t = i | Y, theta)
      # Update alpha
      self._forward()
      denom = 0.0
      for i in range(self.numStates):
        denom += self.alpha[i,t]

      curHiddenStateProbs = np.zeros(self.numStates, dtype="float")
      for i in range(self.numStates):
        curHiddenStateProbs[i] = divide(self.alpha[i,t], denom)

      # P(X_t+1 | X_t) P(X_t) = A[i,j]
      # P(Y_t+1 | X_t+1) = B[i,j]

      nextObservationProbs = np.zeros(self.numCats, dtype="float")
      for k in range(self.numCats):
        for j in range(self.numStates):
          for i in range(self.numStates):
            nextObservationProbs[k] += self.B[j,k]*self.A[j,i]*curHiddenStateProbs[i]

      for v,p in enumerate(nextObservationProbs):
        if self.verbosity > 0:
          print "v,p: ", v,p
        if p >= threshold:
          next_inputs.add(v)

      if len(next_inputs) == 0:
        next_inputs.add(np.argmax(nextObservationProbs))

      return next_inputs