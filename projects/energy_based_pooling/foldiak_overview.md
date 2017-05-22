

# Local Anti-Hebbian learning after P. Földiák

Here is a quick overview of the approach in [[Foldiak]](#foldiak).

**A word of caution:**
Please note that we are **NOT** using the notation from Földiák's paper here (in the paper $W$ denotes the hidden-to-hidden connections).

### Network architecture

Same as for *energy based pooler*. Network connections:

- *Visible-to-hidden* connections encoded by $W$
- *Hidden-to-hidden* connections encoded by $H$. All *hidden-to-hidden* connections are symmetric without self connections,
i.e. we have $h_{ji} = h_{ji}$, and $h_{ii} = 0$ for any $i,j$.
- *Bias* connections encoded by $b$



### Weight updates

 - *Visible-to-hidden*: Hebbian updates $\Delta w_{ij} = \varepsilon_W \ y_i \ (x_j - w_{ij})$
 - *Hidden-to-hidden*: Anti-Hebbian updates defined by
 $\Delta h_{ij} = - \varepsilon_H\ ( y_i \ y_j - s^2)$.
 After any update the connections are clipped to be below $0$, i.e. we set
 $h_{ij} := 0$ if $h_{ij} > 0$.
 - *Bias-to-hidden*: $\Delta b = \varepsilon_B \ (y_i - s)$

Here $s$ denotes a previouly fixed desired unit activation probability (I use $s$ indicating a *sparse* activation probability).



### Encoding inputs

(For a quick overview to this approach cf. for instance the scholarpedia article \[[Hopfield network](http://www.scholarpedia.org/article/Hopfield_network)\] by Hopfield himself)
To compute the hidden vector with a visible input clamped, we solve the following differential equation
$$
	\dot y_i = \sigma\big( W_i x + H_i y - b_i \big) - y_i.
$$
Here we treat $y_i$ as a **continuous** variable.
---*It looks like the ODE computes a flowline/integral curve of the energy gradient*---
Actually, we are looking for a solution of the equation, but are interested in values
such that $\dot y_i = 0$.
Hence we are looking for a fixed point of
$$
f(p) = \sigma\big( Wx + Hp - b \big) .
$$

---*Recall that for the distribution associated to a Hopfield net we have
$P(y_i = 1 \ | \ x) = \sigma\big( W_i x + H_i y - b \big)$. So one would could
think of $y_i$ as the units activation probability*---



# References and Relevant sources

<a name="foldiak">\[1\]</a>
P. Földiák,
*Forming sparse representations by local anti-Hebbian learning*,
Biological Cybernetics 64 (1990), 165-170.

<a name="hopfield">\[2\]</a>
J. J. Hopfield,
*Neurons with Graded Response Have Collective Computational Properties like Those of Two-State Neurons*,
Proc. Natl. Acad. Sci. USA 81 (1984), 3088-3092.

<a name="scholarpedia">\[3\]</a> <http://www.scholarpedia.org/article/Hopfield_network>
