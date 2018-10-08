# Relational Memory Experiments

## CNS 2018 Experiment

The accuracy chart data is computed with the following command:

    python relational_memory_experiment.py --objects 1000 --features 5 --noise 0

The noise is increase up to 16 to generate each point in the chart. The ideal and bag-of-features results are generated with:

    python ideal_observer.py
    python bof.py

## TODO

- Add temporal clustering
    - expect near perfect sensory prediction
- Hotgym
- MNIST with saccading sensor
- MNIST with multiple cortical columns with overlapping RFs
- Add behavior
    - L5 transforms remember the associated motor command
    - Want to drive the L5 transform that is "closest" to destination state
    - Object repr learns weighted values for which L5 transforms lead to it
    - Simple external reward center remembers object state when reward is received and later can drive it
