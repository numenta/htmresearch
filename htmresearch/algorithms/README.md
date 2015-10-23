htmresearch/algorithms
==============

This directory contains the main algorithm implementations we are working
with in research.  Below is a brief description of the primary files.

Note that these are experimental and change all the time. The worst thing
you can do is assume these are done deals and the definitive word on machine
intelligence.

The main files
==========

- `general_temporal_memory.py` - this is a generalization of the HTM temporal
memory algorithm. It contains two new experimental features. 1) It allows
additional external inputs that can connect to the distal dendritic segments.
An example use case is to include external motor input (to simulate motor
efference copy) and perform sensorimotor inference experiments (see
projects/sensorimotor).

- `union_temporal_pooler.py` - this is the most current experimental
implementation of temporal pooling, our third (fourth?) attempt at this.  But
this one is going to stick. Unless it doesn't.


Other files
============

- `faulty_temporal_memory.py` - a subclass of `temporal_memory.py` that
has the ability kill off cells randomly. This was used to test some of the
fault tolerance properties of our Temporal Memory algorithm. The results were
pretty cool - you can read about it in projects/sequence_learning

- `TM.py`, `TM_SM.py`, `spatial_temporal_memory.py`, `temporal_pooler.py` -
older temporal memory, temporal pooling, and "temporal memory for sensorimotor"
implementations. `general_temporal_memory.py` and `union_temporal_pooler.py` has
taken over but some experiments may still rely on this code and we haven't had
time to move them over.

- `hidden_markov_model.py`, `q_learner.py`, `reinforcement_learner.py` -
implementations of other machine learning algorithms for internal testing.

