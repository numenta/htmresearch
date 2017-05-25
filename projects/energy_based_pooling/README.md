

# The HTM Spatial Pooler as a (constrained) Hopfield network

The aim is to express Numenta’s spatial pooler as a Hopfield network. In order to do so we have to express the w-max-overlap procedure as an appropriate energy function. This enables us to extend the energy by an additional term to decorrelate pairwise column activity similar to P. Flödiák’s approach in his 1990’s paper.Relevant keywords are: Hopfield networks, Boltzmann machines, Energy based models, Local anti-Hebbian learning by Flödiák. See the references for relevant and more sources.

<p align="center"><img src="./media/network_architecture_ebp.png"></p>

 - The **new pooler** is implemented as `EnergyBasedPooler` in `energy_based_models/energy_based_pooler.py`.
 - **Brief overview** (still unfinished) of the approach: `latex/notes.pdf`
 - The energy-based pooler is configurable, and can, when configured right, be used as a light weight implementation of the ordinary spatial pooler as well.
 - I present a **new perspective on local inhibition** (this is not related to Hopfield networks), and generalize the local w-max-overlap procedure.
 See Section 6 in `latex/notes.pdf`.
 - First **experimental results** can be found in: `jupyter/results/`. Each folder contains experimental results for a particular setup, briefly described in the associated readme file.
 -  The module also contains an implementation of the approach described in Földiák's paper. A re-run of the "bars experiment" can be found in the notebook `jupyter/Foldiak\ --\ bars.ipynb`.


## References and Relevant sources:

 - P. Földiák, *Forming sparse representations by local anti-Hebbian learning*, Biological Cybernetics 64 (1990), 165–170.
 - Yuwei Cui, Subutai Ahmad, and Jeff Hawkins, *The HTM Spatial Pooler: a neocortical algorithm for online sparse distributed coding*, bioRxiv:085035 (2017).
 - John J. Hopfield, <http://www.scholarpedia.org/article/Hopfield_network>


## First observations and results

##### Energy with *ordinary boosting* only:

<p align="center">
<img src="./jupyter/results/numenta__non_extended__FvuE/ordered_features.png" height="200">
<img src="./jupyter/results/numenta__non_extended__FvuE/reconstruction.png" height="200">
</p>

##### Energy with *extended boosting* decorrelating activity:

<p align="center">
<img src="./jupyter/results/numenta__extended__zero_bounded__AsMS//ordered_features.png" height="200">
<img src="./jupyter/results/numenta__extended__zero_bounded__AsMS/reconstruction.png" height="200">
</p>

##### Energy without size penalty and *extended inhibition* only (non-negative H):

<p align="center">
<img src="./jupyter/results/numenta__extended__no_size_penalty__zero_bounded__8GRh/ordered_features.png" height="200">
<img src="./jupyter/results/numenta__extended__no_size_penalty__zero_bounded__8GRh/reconstruction.png" height="200">
</p>

##### Energy without size penalty (unconstrained H):

<p align="center">
<img src="./jupyter/results/numenta__extended__no_size_penalty__pcPT/ordered_features.png" height="200">
<img src="./jupyter/results/numenta__extended__no_size_penalty__pcPT/reconstruction.png" height="200">
</p>

##### Local inhibition revisited:
Compare Section 6 in `latex/notes.pdf`:

<p align="center">
<img src="./jupyter/results/local_inhibition_revisited__98mf/ordered_features.png" height="200">
<img src="./jupyter/results/local_inhibition_revisited__98mf/reconstruction.png" height="200">
</p>
