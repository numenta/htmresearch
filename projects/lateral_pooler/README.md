

# Dynamically learning lateral inhibitory connections in the HTM Spatial Pooler 


Currently the inhibitory neihbourhoods within the HTM Spatial Pooler (SP) are **hard**-coded.
We extend the SP by introducing lateral inhibitory connections that are learned.
We interpret the lateral weights as a **soft** membership of an inhibitory neighbourhood. Informally a high weight means "being close" and thence a high "competitiveness" between the corresponding units.

Our goal is to reduce the mutual information between output cells by dynamically forming inhibitory connections that force the output units to represent mutually distinct aspects of the inputs. 
In particular we would like to decorrelate the simultaneous activity of distinct ouput cells.

# Where is what

Relevant files and directories:

- `nta/shared-docs/Papers/lateral_pooler/`: Home for the paper and its corresponding latex files. Paper is still under construction...
- `nta/htmresearch/htmresearch/support/lateral_pooler/`: Supporting files such as the function that loads the datasets.

Run

```
python src/run_experiment.py -h
```
to get an example how to use the script.

# Open questions

We are still lacking a convincing use case, that illustrates the differences and strengths of the lateral pooler.


# References and Relevant sources:

 - P. Földiák, *Forming sparse representations by local anti-Hebbian learning*, Biological Cybernetics 64 (1990), 165–170.
 - Yuwei Cui, Subutai Ahmad, and Jeff Hawkins, *The HTM Spatial Pooler: a neocortical algorithm for online sparse distributed coding*, Front. Comput. Neurosci. (2017), 111.
 - ...
 