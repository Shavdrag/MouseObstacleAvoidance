Repo for the Obstacle Avoidance by Deep Reinforcement Learning.


The main files are the ExploreEnv.py and ExploreMain.py, representing the environment and the main script.

The Controller.py and ToSim.py is slightly modified code provided by TU Munich and is the controller and the communicationscript for mujoco simulation.

Maze_generator.py, xml_whisker_generator.py and utilities.py are for generating random mazes, generate the whiskers in mujoco and utility functions (e.g. for ploting).


The different directorys are models and trained_nets, that contain the mujoco models including stl files and the trained neural nets accordingly. Additionally there is the LegModel, that also is provided by TU Munich and calculates the motor position for the next step of the robot.
