from __future__ import division
import numpy as np

# function [G_I_EL,G_I_ER,G_EL_I,G_ER_I,G_I_I]
def compute_hardwired_weights(rho,N_E,N_I,periodic, onlyI=False):
  '''
  %This function returns the synaptic weight matrices
  %(G_I_EL,G_I_ER,G_EL_I,G_ER_I,G_I_I) and the suppressive envelope
  %(A_env), based on:
  %
  % - the scale of the synaptic profiles (rho)
  % - the size of the exctitatory and inhibitory pops (N_E, N_I)
  % - the boundary conditions of the network (periodic=1 for periodic b.c.s; periodic = 0 for aperiodic b.c.s)

  %The parameters below are arranged according to the following order of
  %synaptic weights: EL->I, ER->I, I->EL, I->ER, I->I
  % (see Supplementary Methods of PNAS paper for description of params below)

  It was ported directly from code provided by Ila Fiete.
  '''

  weight_sizes = np.asarray([[N_I,N_E], [N_I,N_E], [N_E,N_I], [N_E,N_I], [N_I,N_I]])
  gamma_param = np.asarray([N_I/N_E, N_I/N_E, N_E/N_I, N_E/N_I, N_I/N_I])
  eta_param = np.asarray([1.5*21, 1.5*21, 8, 8, 24])     #controls overall strength of weights
  epsilon_param = np.asarray([0, 0, 0, 0, 1])            #controls contribution of negative gaussian in diff. of Gaussians weights
  sigma_param = np.asarray([2, 2, 5, 5, 3])              #controls width of weight profiles
  Delta_param = np.asarray([-2, 2, 8, -8, 3])            #controls weight asymmetries
  mu_param = np.asarray([0, 0, -1, 1, 0])                #controls weight asymmetries
  delta_param = np.asarray([0, 0, 3, 3, 3])            #controls weight asymmetries

  #the for-loop below iterates through the 5 synaptic weight types
  for k in [4,3,2,1,0]:


      #N_2 = size of projecting pop; N_1 = size of receiving pop.
      N_1 = weight_sizes[k][0]
      N_2 = weight_sizes[k][1]

      #create envelopes based on pop. sizes
      A_1 = create_envelope(periodic,N_1)[0]
      A_2 = create_envelope(periodic,N_2)[0]

      #Create synaptic weight matrix
      G = np.zeros((N_1, N_2))
      for i in range(N_1):
          for j in range(N_2):

              x = i - gamma_param[k]*j

              c_left = min(N_1 - np.abs(np.mod(x - Delta_param[k], N_1)), np.abs(np.mod(x - Delta_param[k], N_1)))
              c_right = min(N_1 - np.abs(np.mod(x + Delta_param[k], N_1)), np.abs(np.mod(x + Delta_param[k], N_1)))
              c_0 = min(N_1 - np.abs(np.mod(x, N_1)), np.abs(np.mod(x, N_1)))

              G[i, j] = eta_param[k]/rho*A_1[i]*A_2[j]*((c_0-delta_param[k]*rho) >= 0)*(((-mu_param[k]*x) >= 0)*((mu_param[k]*(x+mu_param[k]*N_1/2)) >= 0) +
                           ((mu_param[k]*(x-mu_param[k]*N_1/2)) >= 0))*(np.exp(-c_left**2/(2*(sigma_param[k]*rho)**2)) +
                           epsilon_param[k]*np.exp(-c_right**2/(2*(sigma_param[k]*rho)**2)))

      if k==0:
          G_I_EL = G
      elif k==1:
          G_I_ER = G
      elif k==2:
          G_EL_I = G
      elif k==3:
          G_ER_I = G
      else:
          G_I_I = G
          if onlyI:
            return G_I_I, G_I_I, G_I_I, G_I_I, G_I_I

  return G_I_EL, G_I_ER, G_EL_I, G_ER_I, G_I_I


#function A
def create_envelope(periodic,N):
  '''
    %This function returns an envelope for network of size N; The envelope can
    %either be suppressive (periodic = 0) or flat and equal to one (periodic = 1)
  '''
  kappa = 0.3; # controls width of main body of envelope
  a0 = 30;    # contrls steepness of envelope

  if periodic==0:
      A = np.zeros(N)
      for m in range(N):
          r = np.abs(m-N/2);
          if r<kappa*N:
              A[m] = 1
          else:
              A[m] = np.exp(-a0*((r-kappa*N)/((1-kappa)*N))**2)
  else:
      A = np.ones((1,N));

  return A
