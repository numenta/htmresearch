

Inputs
------

 - Input type       : (x,y,y,y)
 - bits per axis    : 50
 - weight per axis  : 5
 - number of streams: 4

Learning
--------

Minibatches of size 20

Energy-based Pooler
---------------

Configuration...

 - input size  : 200
 - output size : 128
 - code weight : 4
 - sparsity    : 0.03125
 - Boost_b     : 100.0
 - Boost_H     : 0.0
 - incr, decr  : 0.01, 0.01
 - LR_hidden   : 0.01
 - LR_bias     : 0.01

Docstrings...

**initialize connections**
Randomly initializes the visible-to-hidden connections.

Each weight is independently sampled from a uniform distribution $U(0,1)$.
The weights are NOT normalized!

**energy:**
Numenta's energy:
$$
    E(x,y)  = - \sum_i y_i \cdot \exp( - b_i ) \cdot (\sum_j W_{ij} \ x_j ) + S(y)
$$
where the size size penalty is given by
$$
    S(y) = \begin{cases}
            0        &  \text{if $\|y\| \leq w$, and} \\
            +\infty  &  \text{otherwise,}
           \end{cases}
$$

**update weights:**
Method that updates the network's connections. The
Weights are updated and computed according to

 - Visible to hidden: $ \Delta W_{ij} = y_i \cdot  ( \varepsilon_{\small{+}} \ x_j - \varepsilon_{\small{-}} \ \bar x_j ) $
 - Bias:              $ b_{i}  = B_b \cdot \alpha_i $
 - Hidden to hidden : $ H_{ij} = B_H \cdot ( \alpha_{ij} - \alpha_i \ \alpha_j ) $


**find energy minimum:**
Naive hill descend algorithm:
Starts at $y=0$ and iteratively chooses the direction of steepest descent from
a neighbourhood of the current position. If not indicated otherwise the neighbourhood
consists of elements at Hamming distance less than or equal to $1$.

---------------
