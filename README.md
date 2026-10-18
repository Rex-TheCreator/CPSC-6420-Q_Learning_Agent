# Training a Q-Learning agent using Gymnasium API
## CPSC 6420 Assignment 2 

Follows the Assignment 2 Instructions which go through requirements for building and evaluating a Q-Learning Agent

Summary:
- Frozen Lake was selected
- Modified Epsilon, Gamma, Alpha to alter agent performance
- Conducted tests to find the best performing agent
- Added noise(stochasticity) to the environment

Directory:
- results\                      contains images from different tests
- agent.py                      main script for agent
- CPSC 6420 Assignment 2.pdf    Report of findings according to assignment directions
- inspect_frozen_lake.py        script for understanding Frozen Lake enviroment
- CPSC4420_HW2_F26.pdf          Assignment directions
- frozenlake-agent-episode-0    video of an episode of the trained agent. Created by agent.py

Environment:
- conda was used as the package manager for this project for python 3.11
- numpy, matplotlib, and gymnasium\[toy-text\] must be installed, moviepy can be optionally installed for video functionality. 'pip install numpy matplotlib gymnasium[toy-text] moviepy'
- agent.py is the main script and is run by 'python agent.py'
- to create a video of the trained agent, uncomment the code block in agent.py
