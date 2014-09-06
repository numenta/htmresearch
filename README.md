Open Research?
==============

We have released all our commercial HTM algorithm code to the open source 
community within NuPIC. We use that code in our product. The NuPIC
open source community and Numenta continues to maintain and improve that 
regularly. 

Internally though we continue to evolve and expand the ideas towards a 
full blown cortical framework. Those research ideas are constantly in flux as
we tweak and experiment. To go along with that we have a separate experimental 
codebase that sits on top of NuPIC.   

We get a lot of questions about it. As such we wondered whether it is 
possible to be even more open about that work.  Could we release our day 
to day research code in a public repository? Would people get confused?  
Would it slow us down?

We discussed these tradeoffs on the NuPIC mailing list. Based on that 
discussion, we decided to go ahead and create `nupic.research` It contains 
the code for experimental algorithm work done internally at Numenta.

The code includes prototypes and experiments with different algorithm 
implementations. This is all temporary, ever-changing experimental code, 
which poses some challenges.

Hence the following **DISCLAIMERS**:

 
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

