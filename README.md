INTRODUCTION
This repository contains the code for experimental algorithm work done internally at Numenta. A description of that research is <link: https://www.numenta.com/neuroscience-research/> available here </link>.

Open Research?
==============

We have released all our commercial HTM algorithm code to the open source community within NuPIC. The NuPIC open source community continues to maintain and improve that regularly (see * https://discourse.numenta.org for discussions on that codebase. Internally we continue to evolve our theory towards a full blown cortical framework.

We get a lot of questions about it and we wondered whether it is possible to be even more open about that work. Could we release our day to day research code in a public repository? Would people get confused? Would it slow us down?

We decided to go ahead and create htmresearch. It contains experimental algorithm code done internally at Numenta. The code includes prototypes and experiments with different algorithm implementations. It sits on top of NuPIC and requires you have NuPIC installed.

Our research ideas are constantly in flux as we tweak and experiment. This is all temporary, ever-changing experimental code, which poses some challenges. Hence the following DISCLAIMERS:

 
What you should understand about this repository
================================================

- the contents can change without warning or explanation
- the code will change quickly as experiments are discarded and recreated
- it might not change at all for a while
- it could just be plain wrong or buggy for periods of time
- code will not be production-quality and might be embarrassing
- comments and questions about this code may be ignored
- Numenta is under no obligation to properly document or explain this
codebase or follow any understandable process
- repository will be read-only to the public
- if we do work with external partners, that work will probably NOT be here
- we might decide at some point to not do our research in the open anymore and 
instead delete the whole repository

We want to be as transparent as possible, but we also want to move
fast with these experiments so the finalized algorithms can be
included into NuPIC as soon as they are ready. We really hope this repository
is helpful and does not instead create a lot of confusion about what's coming
next.  


Installation
============

OK, enough caveats. Here are some installation instructions though mostly you
are on your own. (Wait, was that another caveat?)

## Released Version ![](https://img.shields.io/pypi/v/htmresearch.svg)

    pip install nupic htmresearch

## Developer

Requirements:

- `nupic` and `nupic.core`
  - `pip install nupic --user` should be sufficient
- `htmresearch-core`
  - To install, follow the instructions in the
    [htmresearch-core README](https://github.com/numenta/htmresearch-core).
- Various individual projects may have other requirements. We don't formally
  spell these out but two common ones are pandas and plotly.

Install using setup.py like any python project. Since the contents here change
often, we highly recommend installing as follows:

    python setup.py develop --user

After this you can test by importing from htmresearch:

    python
    from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory

If this works installation probably worked fine. BTW, the above class is a
modified version of TemporalMemory that we are currently researching. It
includes support for feedback connections (through apical dendrites) and
external distal basal connections.

You can perform a more thorough test by running the test script from the repository root:

    %> ./run_tests.sh 


Archive
=======

Some of our old research code and experiments are archived in the following repository: 
 
* https://github.com/numenta-archive/htmresearch

