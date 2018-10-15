# Multi-column experiment

Create a network consisting of multiple columns to run object recognition 
experiments. 

Each column contains one L2, one L4 and one L6a layers. In addition all the L2 
columns are fully connected to each other through their lateral inputs.

```
                            +----lateralInput--+
                            | +--------------+ |
                            | |       +1     | |
 Phase                      v |              v |
 -----                   +-------+         +-------+
                  reset  |       |         |       | reset
 [3]              +----->|  L2   |         |  L2   |<----+
                  |      |       |         |       |     |
                  |      +-------+         +-------+     |
                  |        |   ^             |   ^       |
                  |     +1 |   |          +1 |   |       |
                  |        |   |             |   |       |
                  |        v   |             v   |       |
                  |      +-------+         +-------+     |
 [2]        +----------->|       |         |       |<----------+
            |     |      |  L4   |         |  L4   |     |     |
            |     +----->|       |         |       |<----+     |
            |     |      +-------+         +-------+     |     |
            |     |        |   ^             |   ^       |     |
            |     |        |   |             |   |       |     |
            |     |        |   |             |   |       |     |
            |     |        v   |             v   |       |     |
            |     |      +-------+         +-------+     |     |
            |     |      |       |         |       |     |     |
 [1,3]      |     +----->|  L6a  |         |  L6a  |<----+     |
            |     |      |       |         |       |     |     |
            |     |      +-------+         +-------+     |     |
       feature  reset        ^                 ^      reset  feature
            |     |          |                 |         |     |
            |     |          |                 |         |     |
 [0]     [sensorInput]  [motorInput]      [motorInput] [sensorInput]
```


Use the following command to run all **L2-L4-L6a** experiments:
```
python multi_column_convergence.py -c experiments_l2l4l6a.ini
```
The [./results](./results) directory will contain the the chart and raw results 
data used to create the chart. One result folder per experiment. 
See [experiments_l2l4l6a.ini](experiments_l2l4l6a.ini) for details on the 
parameters used by the experiments.


--------------------------------------------------------------------------------
> This experiment is based on [PyExperimentSuite](https://github.com/rueckstiess/expsuite). 
> We keep a local copy in `htmresearch/support/expsuite.py`. For more details on 
> how to configure the experiments see [PyExperimentSuite Documentation](https://github.com/rueckstiess/expsuite/blob/master/documentation.pdf)  