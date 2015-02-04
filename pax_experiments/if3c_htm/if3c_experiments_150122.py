# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:14:22 2015

@author: pfrady
"""



from pylab import *
from brian2 import *

#%% 3 Compartment Excitatory neuron params

C_a = .40 * nF # axon capacitance
C_s = .80 * nF # coma capacitance
C_d = .60 * nF # dendrite capacitance

g_c_ds = 55.0 * nS # dendrite-soma coupling conductance
g_c_sa = 55.0 * nS # soma-axon coupling conductance

g_l_a = 2.0 * nS # leak axon
g_l_s = 10.0 * nS # Leak soma
g_l_d = 3.0 * nS # Leak dendrite

V_t = -58.0 * mV # Threshold
V_r = -62.0 * mV # Reset

E_l = -60.0 * mV # Rest

E_i_s = -60.0 * mV # Inhibition reversal potential soma
E_i_d = -60.0 * mV # Inhibition reversal potential dendrite

tau_I_d = 1 * ms
tau_I_s = 1 * ms

tau_g_i_d = 25 * ms
tau_g_i_s = 25 * ms



#%% Neuron Equations

# This 3c neuron can do additive operations through I_s, I_d, and can do multiplicative
# shunting inhibition by operating on g_i_s, g_i_d
eqs_3c = Equations('''
    dVa/dt = (g_l_a * (E_l - Va) + g_c_sa * (Vs - Va)) / C_a : volt    
    dVs/dt = (g_l_s * (E_l - Vs) + g_c_sa * (Va - Vs) + g_c_ds * (Vd - Vs)
             + g_i_s * (E_i_s - Vs) + I_s + I_ext) / C_s : volt
    dVd/dt = (g_l_d * (E_l - Vd) + g_c_ds * (Vs - Vd) 
             + g_i_d * (E_i_d - Vd) + I_d) / C_d : volt
    dg_i_s/dt = -g_i_s / tau_g_i_s : siemens
    dg_i_d/dt = -g_i_d / tau_g_i_d : siemens
    dI_s/dt = -I_s / tau_I_s : amp
    dI_d/dt = -I_d / tau_I_d : amp
    I_ext : amp
''')

## So this was necessary because I was inputting Timed arrays as external 
## inputs. But I only wanted to put inputs into exc, and not into the inh cells
## These equations have the I_ext_s(t), which refers to the timed arrays.
eqs_3c_in = Equations('''
    dVa/dt = (g_l_a * (E_l - Va) + g_c_sa * (Vs - Va)) / C_a : volt    
    dVs/dt = (g_l_s * (E_l - Vs) + g_c_sa * (Va - Vs) + g_c_ds * (Vd - Vs)
             + g_i_s * (E_i_s - Vs) + I_s + I_ext_s(t)) / C_s : volt
    dVd/dt = (g_l_d * (E_l - Vd) + g_c_ds * (Vs - Vd) 
             + g_i_d * (E_i_d - Vd) + I_d + I_ext_d(t)) / C_d : volt
    dg_i_s/dt = -g_i_s / tau_g_i_s : siemens
    dg_i_d/dt = -g_i_d / tau_g_i_d : siemens
    dI_s/dt = -I_s / tau_I_s : amp
    dI_d/dt = -I_d / tau_I_d : amp
''')

#%% Synapse Equations

# Equations for lateral/recurrent excitatory connections
#eqs_lat_model = Equations('''w : amp''')
eqs_lat_model = '''w : amp'''

eqs_lat_pre = '''I_d += w'''

# Equations for shunting inhibition at the soma
#eqs_inh_model = Equations('''w : siemens''')
eqs_inh_model = '''w : siemens'''

#eqs_inh_pre = Equations('''g_i_s += w''')
eqs_inh_pre = '''g_i_s += w'''


#%% Create the Neuron Groups

N_cols = 5 # Number of mini columns
N_exc_p_col = 10 # Number of excitatory cells per mini column
N_exc = N_cols * N_exc_p_col

N_PV = 5

G_exc = NeuronGroup(N_exc, eqs_3c_in, threshold='Va > V_t', reset='Va = V_r')
G_PV = NeuronGroup(N_PV, eqs_3c, threshold='Va > V_t', reset='Va = V_r')
G_M = NeuronGroup(N_cols, eqs_3c, threshold='Va > V_t', reset='Va = V_r')


#%% Make the stimulus

run_time = 1 * second
input_dt = 20 * ms


# so we want every N_exc_p_col neurons to have the same stimulus

stimulus = rand(run_time // input_dt, N_cols)
stim_cols = tile(stimulus, (1, N_exc_p_col))

I_ext_s = TimedArray(stim_cols * nA, dt=input_dt)

# Just making this 0
I_ext_d = TimedArray(0 * stim_cols * nA, dt=input_dt)

# So the way it is set up now is that every N_cols neurons are in the same column

#%% Set up the connectivity

# The PV cels just receive input from everything
S_exc_pv = Synapses(G_exc, G_PV, model=eqs_lat_model, pre=eqs_lat_pre)

# Excitatory neurons have recurrent connections
S_exc_exc = Synapses(G_exc, model=eqs_lat_model, pre=eqs_lat_pre)

# PV does shunting gain control
S_pv_exc = Synapses(G_PV, model=eqs_inh_model, pre=eqs_inh_pre)

# M cells are connected to their column
S_exc_M = Synapses(G_exc, G_M, model=eqs_lat_model, pre=eqs_lat_pre)
#S_M_exc = Synapses(G_M, G_exc, model=eqs_lat_model, pre=eqs_lat_pre)
S_M_exc = Synapses(G_M, G_exc, model=eqs_inh_model, pre=eqs_inh_pre)

S_exc_pv.connect(True) # All excitatory to PV with prob of 0.9
S_pv_exc.connect(True) # All pv to exc with prob of 0.9

S_exc_exc.connect(True, p=0.5)


# Ok, want just neurons in a column to connect to M, and vice-versa
S_M_exc.connect('i == (j % N_cols)')
S_exc_M.connect('j == (i % N_cols)')


#%% Monitors

SpM_exc = SpikeMonitor(G_exc)
StM_exc = StateMonitor(G_exc, True, True)

SpM_PV = SpikeMonitor(G_PV)
StM_PV = StateMonitor(G_PV, True, True)
SpM_M = SpikeMonitor(G_M)
StM_M = StateMonitor(G_M, True, True)


#%% Initialize

G_exc.Va = E_l * ones(N_exc)
G_PV.Va = E_l * ones(N_PV)
G_M.Va = E_l * ones(N_cols)
G_exc.Vs = E_l * ones(N_exc)
G_PV.Vs = E_l * ones(N_PV)
G_M.Vs = E_l * ones(N_cols)
G_exc.Vd = E_l * ones(N_exc)
G_PV.Vd = E_l * ones(N_PV)
G_M.Vd = E_l * ones(N_cols)

G_M.I_ext = 0.2 * nA
G_PV.I_ext = 0.3 * nA

S_exc_pv.w = 0.2 * nA
S_exc_exc.w = 0.002 * nA
S_pv_exc.w = 15.0 * nS
S_exc_M.w = 0.5 * nA
S_M_exc.w = 40.0 * nS
#S_M_exc.w = -0.5 * nA

#%% Run

run(run_time)


#%% Plot

plot(SpM_exc.it[1], SpM_exc.it[0], '|')

#%% PLot excitatory raster

figure(1)
clf()
subplot(211)
colors = get_cmap('Set3', N_cols)


for idx in range(N_cols):
    col_idx = N_cols * arange(N_exc_p_col) + idx
    
    for jdx in range(len(col_idx)):
        c_idx = SpM_exc.it[0] == col_idx[jdx]
        plot(SpM_exc.it[1][c_idx], (jdx+idx*N_exc_p_col) * ones((sum(c_idx))), '|', c=colors(idx))


subplot(212)

for idx in range(stimulus.shape[1]):
    
    plot(stimulus[:,idx] + idx, c=colors(idx))

#%%

figure(2)
clf()
subplot(211)
plot(SpM_PV.it[1], SpM_PV.it[0], '|')

subplot(212)
plot(SpM_M.it[1], SpM_M.it[0], '|')

#%%

figure(3)
clf()
plot(StM_exc[0].Vs)




#%% plot 1 column, and the M

idx = 0


col_idx = N_cols * arange(N_exc_p_col) + idx

figure(4)
clf()
subplot(411)    
for jdx in range(len(col_idx)):
    c_idx = find(SpM_exc.it[0] == col_idx[jdx])
    plot(SpM_exc.it[1][c_idx], (jdx+idx*N_exc_p_col) * ones((len(c_idx))), '|', c=colors(idx))
title('Col %d spikes' % idx)


subplot(412)
plot(StM_exc[0].t, StM_exc[0].Vs)
plot(StM_exc[0].t, StM_exc[0].Va)

title('Excitatory')

subplot(413)
plot(StM_M[0].t, StM_M[0].Vs)
plot(StM_M[0].t, StM_M[0].Va)
title('M')

subplot(414)
plot(StM_PV[0].t, StM_PV[0].Vs)
plot(StM_PV[0].t, StM_PV[0].Va)
title('PV')





