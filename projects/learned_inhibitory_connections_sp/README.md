

# Dynamically learning lateral inhibitory connections in the HTM Spatial Pooler 


Currently the inhibitory neihbourhoods within the HTM Spatial Pooler (SP) are **hard**-coded.
We extend the SP by introducing lateral inhibitory connections that are learned.
We interpret the lateral weights as a **soft** membership of an inhibitory neighbourhood. Informally a high weight means "being close" and thence a high "competitiveness" between the corresponding units.

Our goal is to reduce the mutual information between output cells by dynamically forming inhibitory connections that force the output units to represent mutually distinct aspects of the inputs. 
In particular we would like to decorrelate the simultaneous activity of distinct ouput cells.


## References and Relevant sources:

 - P. Földiák, *Forming sparse representations by local anti-Hebbian learning*, Biological Cybernetics 64 (1990), 165–170.
 - Yuwei Cui, Subutai Ahmad, and Jeff Hawkins, *The HTM Spatial Pooler: a neocortical algorithm for online sparse distributed coding*, Front. Comput. Neurosci. (2017), 111.
 - ...
 