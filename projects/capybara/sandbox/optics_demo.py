"""
Install: 
    git clone https://github.com/annoviko/pyclustering
    cd pyclustering
    pip install . --user
"""

from pyclustering.cluster.optics import optics, ordering_analyser, ordering_visualizer 
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample

# Read sample for clustering from some file
# 2D matrix
sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)

# Run cluster analysis where connectivity radius is bigger than real
radius = 2.0
neighbors = 3

# Create OPTICS algorithm for cluster analysis
optics_instance = optics(sample, radius, neighbors)

# Run cluster analysis
optics_instance.process()

# Obtain results of clustering
clusters = optics_instance.get_clusters()
noise = optics_instance.get_noise()

# Obtain reachability-distances
ordering = ordering_analyser(optics_instance.get_ordering())

# Visualization of cluster ordering in line with reachability distance.
ordering_visualizer.show_ordering_diagram(ordering)

