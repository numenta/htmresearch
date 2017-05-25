import numpy as np


def numenta_local_wmax_extended(self, x):
    r"""
    New local w-max procedure (experimental).

    This `encode`-function extends the local w-max overlap procedure.
    See ``Numenta’s local inhibition revisited'' (Section 6) in `latex/notes.pdf`.
    Note that we the ``activation probabilies'' $a_{ij}$ are already encoded as $h_{ij}$.
    """
    pooler = self
    W      = pooler.connections.visible_to_hidden
    H      = pooler.connections.hidden_to_hidden
    b      = pooler.connections.hidden_bias
    n, m   = pooler.output_size, pooler.input_size
    w      = pooler.code_weight
    s      = pooler.sparsity
    y      = np.zeros(n)

    a      = pooler.average_activity

    scores = np.exp(-b)*np.dot(W,x)

    for i in range(n): 

        estimated_excitement = 0.
        for j in range(n):

            # a_ij = a[i,j]/a[i,i]
            a_ij = H[i,j]

            if scores[j] >= scores[i]:
                estimated_excitement += a_ij

        if estimated_activity < s:
            y[i] = 1.

    return y


def energy_minimum(self, x):
    """
    Given an input array `x` it returns its associated encoding `y(x)`, that is, 
    a stable configuration (local energy minimum) of the hidden units 
    while the visible units are clampled to `x`.
    """
    E      = self.energy
    y_min  = self.find_energy_minimum(E, x)
    return y_min



