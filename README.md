# EPL FlappyBird

This repository contains the code for a master thesis for UCLouvain. 
The aim of the project was to train DQN agent to play FlappyBird and add some enviornmenta modifitication to see how the training was impacted.
Among thoses modifications we can find:
- Training on fixed or random pipes
- Training with some randomness in the Jump Force
- Training with increased action space

The code is based on the following sources:
- for the FlappyBird game: https://github.com/Talendar/flappy-bird-gym
- for the DQN agent: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

The code is structured as follows in the source directory:
- flappy_bird_gym: contains the FlappyBird game and the environment modifications
- run_baseline_example: Example of the output of a training
- agent_simple, agent_multi, agent_RGB: DQN agents with some modifications (simple is the classic agent, RGB is the agent for RGB images, multi is the agent with increased action space)
- evaluation: code to evaluate the results of the training and create the plots
- exp_X: code to run the experiments
- exp_X_test: code to test the experiments
- utils: some utility functions