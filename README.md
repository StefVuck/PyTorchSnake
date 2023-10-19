# PyTorch-based Snake Game
Reinforcement Machine Learning Snake Game using PyTorch framework

### Overview: 
This is my first main large machine learning project, using the PyTorch framework.

The actual snake game is an adapted version of assets from https://github.com/patrickloeber/python-fun/tree/master/snake-pygame and it was very helpful to have a easy framework to build off.
However the controls were made for a human to play and thus the game was adapted. 

### Machine Learning Information:
I'm using Deep Q-Network with PyTorch, a form of reinforcement learning, and I'm specificially implementing the Bellman Equation: 
$` Q_{new} = r + \gamma * Q_{next\_state} `$ where r is the most immediate reward, and $`\gamma`$ is our discount factor which just changes our weighting of the $`Q_{next\_state}`$ in this case

### Tasks and Goals:
##### Main Goals:
- ~~Implement Game to be controlled by Agent~~
- ~~Implement Agent~~
- ~~Implement Model~~
- ~~Implement Saving of Model~~
- Implement loading of saved models
- Experiment with different parameters

##### Stretch Goals:
- Create a way for the model to play without using the comparatively slow pygame
- Use cuda tp use the GPU for processiong, hopefully with the last point making 1000 generations not take 4+ hours
- 10,000 generations of learning for a few different parameters?
