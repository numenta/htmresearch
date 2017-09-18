
## Untangling Sequences: Behavior vs. External Causes

This directory contains Python scripts that run sensorimotor sequences
in combination with pure temporal sequences.

The goal is to test whether a single neural mechanism can
automatically discover what parts of a changing sensory stream are due
to movement and which parts are due to external causes, and to learn
predictive models of both types of causes simultaneously using simple
learning rules.

## ABSTRACT (FROM PAPER IN PROCESS)

There are two fundamental reasons why sensory inputs to the brain change
over time. Sensory inputs can change due to external factors or they can
change due to our own behavior. Interpreting behavior-generated changes
requires knowledge of how the body is moving, whereas interpreting
externally-generated changes relies solely on the temporal sequence of
input patterns. The sensory signals entering the neocortex change due to
a mixture of both behavior and external factors. The neocortex must
disentangle them but the mechanisms are unknown. In this paper, we show
that a single neural mechanism can learn and recognize both types of
sequences. In the model, cells are driven by feedforward sensory input
and are modulated by contextual input. If the contextual input includes
information derived from efference motor copies, the cells learn
sensorimotor sequences. If the contextual input consists of nearby
cellular activity, the cells learn temporal sequences. Through
simulation we show that a network containing both types of contextual
input automatically separates and learns both types of input patterns.
We review experimental data that suggests the upper layers of cortical
regions contain the anatomical structure required to support this
mechanism.

## USAGE
To run one of the experiments, say the one for Figure 4A, and generate
the appropriate plot:

    python combined_sequences.py 4A
    python generate_plots.py

To see the various options:

    python combined_sequences.py -h
    python generate_plots.py -h

Note: the results may not be identical to the charts in the paper, due
to possible changes in the order the random number generator is called
and perhaps also algorithm changes.  They should be similar though, and
conclusions and takeaways should be the same.

Another note: a couple of the experiments, 5B in particular, take a long
time to run (5B takes several hours).  The script will use all available
cores on your machine, so beware.


