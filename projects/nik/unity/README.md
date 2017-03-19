Numenta Inverse Kinematics Simulation
=====================================

This project uses a simple 2D robot simulation created in Unity to test **Numenta Inverse Kinematics (NIK)**

## Setup

- Download and install [Unity](http://www.unity3d.com)
- Make sure the correct versions for python and nupic are installed and available from your path.
- Copy your pre-trained serialized HTM model to [Assets/Resources/Python/](Assets/Resources/Python/)
- Link [../nik_htm.py](../nik_htm.py) to [Assets/Resources/Python/nik_htm.py.txt](Assets/Resources/Python/nik_htm.py.txt)

        cd Assets/Resources/Python
        ln -s ../../../../nik_htm.py nik_htm.py.txt

> NOTE: Whenever you update `nik_htm.py` you need to delete the cached copy made by Unity in the application data directory `~/Library/Application Support/Numenta/Nik/Python`

## Controlling the Robot

- **Left Arm**
    - `Q` : Shoulder Up
    - `A` : Shoulder Down
    - `W` : Elbow Up
    - `S` : Elbow Down

- **Right Arm**
    - `P` : Shoulder Up
    - `L` : Shoulder Down
    - `O` : Elbow Up
    - `K` : Elbow Down

- **Robot Behaviors**
    - `R` : Move both arms to Rest pose (0, 180)
    - `N` : Move robot and balloons to "Numenta" pose (60, 60)
    - `Space` : Move robot and balloons to a random pose

- **Nik Behaviors**
    - `T` : Start training
    - `0` : Start inverse kinematics
    - `1` : Move robot and balloons predetermined pose 1
    - `2` : Move robot and balloons predetermined pose 2
    - `3` : Move robot and balloons predetermined pose 3
    - `D` : Print debug information to unity console

## Experiment Parameters

To change the experiment parameters you need to select the robot arm (`Arm_L`) from Unity's `Scene View` and update the parameters on the `Inspector View`. See [Experiment1.cs](Assets/Scripts/Experiment1.cs) for details.

Here are the available parameters:

- `shoulderResolution` : Shoulder angle step resolutions in degrees
- `elbowResolution` : Elbow angle step resolutions in degrees
- `trainingSize` : Total random points to use during training
- `saveLog` : Whether or not to save the training data log. The data will be stored in  `~/Library/Application Support/Numenta/Nik/data/`
- `useNik` : Whether or not to Nik for inverse kinematics
- `serializedModelName` : Pre-trainded serialized model name to use. Path is relative to the `Assets` folder
- `target` : Object to seek using inverse kinematics
- `seed` : Random number generator seed
