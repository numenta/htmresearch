# Learning Grid Cells

## V2 - Dordek, et. al.

Reproduced something similar to the Dordek paper calculations using PCA in `dordek.py`. Working on simulations of the Hebbian-learning variant in `dordek_sim.py`.

## V1 - Motor Mappings and Grid Cells

The `engineered.py` file involved using sensory information to anchor a location and then used various strategies for learning connections between grid cells and also learning motor mappings. This was not too successful. The biggest issue was learning the wrap around of the grid cells. The motor mapping was also difficult to learn in an online matter due to the high frequency of local maxima in the constraint satisfaction formulation of the problem.
