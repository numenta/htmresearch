

# The HTM Spatial Pooler as a Hopfield network

 - The **new pooler** is implemented as `EnergyBasedPooler` in `energy_based_models/energy_based_pooler.py`. 
 - **Brief overview** (still unfinished) of the approach: `latex/notes.pdf` 
 - First **experimental results** can be found in: `jupyter/results/`. Each folder contains experimental results for a particular setup, briefly described in the associated readme file.
 -  The module also contains an implementation of the approach described in Földiák's paper. A re-run of "bars" experiment can be found in the notebook `jupyter/Foldiak\ --\ bars.ipynb`.


<p align="center"><img src="./media/network_architecture_ebp.png"></p>


##### References and Relevant sources:

 - P. Földiák, *Forming sparse representations by local anti-Hebbian learning*, Biological Cybernetics 64 (1990), 165–170.
 - Yuwei Cui, Subutai Ahmad, and Jeff Hawkins, *The HTM Spatial Pooler: a neocortical algorithm for online sparse distributed coding*, bioRxiv:085035 (2017).
 - John J. Hopfield, <http://www.scholarpedia.org/article/Hopfield_network>


# First observations

#### 1. Ordinary Boosting *vs.* Extended Boosting

Compare *ordinary boosting*
with *extended boosting* decorrelating activity:

```
open jupyter/results/three-to-one__non_extended/reconstruction.mp4
open jupyter/results/three-to-one__extended/reconstruction.mp4
```

Additionally normalized the random initial weights:

```
open jupyter/results/three-to-one__extended__normalized/reconstruction.mp4
```
#### 2. Without size penalty

If size penalty removed from the energy, compare results from E with *extended boosting-inhibition* term
to E with *extended inhibition only* (ie. ensure a non-negative H)

```
open jupyter/results/three-to-one__no_size_peanlty__b5sj/reconstruction.mp4
open jupyter/results/three-to-one__zero_bound__no_size_penalty__ibAj/reconstruction.mp4

```

