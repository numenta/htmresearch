NIK - Numenta Inverse Kinematics
================================

This project was done as part of an internal company hackathon on March
13-14, 2017.

Luiz and Subutai joined forces to explore whether we can learn [inverse
kinematics](https://en.wikipedia.org/wiki/Inverse_kinematics) using HTM.
This is a concept well known in robotics that computes the desired
position of a figure.  It’s essentially figuring out: if I know the
position I want to move to, how do I position the body to get there? For
example, if I want to reach out and touch something at a known location,
how would I position my arm so that my fingers are actually touching the
object?  The brain has to solve the same problem, so they wanted to
figure out if it is possible to learn inverse mappings using
biologically plausible HTM neurons.

Learning inverse kinematics is a challenging task in general. The
results are preliminary, and there are problems with this hack, but
after two days of hacking, all signs point to, “Yes, you can use HTM to
learn inverse kinematics.” The overall project and the issues we ran
into are described in the presentation below.


Contents
========

We used Unity to display the graphics and python to train and run the
HTM.

* hello_ik.py
    * A simple "hello world" program we used to figure out how to feed
      data to the HTM so it could learn a mapping.

* nik_htm.py
    * Python code for training HTM and running inference. In order to
      interface with Unity, it is written as a program that continually
      accepts inputs and commands from stdin, and outputs desired
      body configurations to stdout.  Debugging output is sent to
      stderr.

* nik_analysis.py
    * A script that analyzes the histogram of training points

* unity\
    * Directory containing code for displaying the robot NIK

* presentation\
    * End of hackathon presentation made internally by Luiz and Subutai

* data\train50k_seed37_2.csv
    * The main training file we ended up using

* data\test10K_2.csv
    * The main test file we ended up using

