# Udacity Deep Reinforcement Nanodegree Project 3: Collaboration and competition
This repository contains implementation of Collaboration and Competition project as a part of Udacity's Deep Reinforcement Learning Nanodegree program.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Install Python3.

2. [Optional] Create (and activate) a new environment with Python 3.x.
	
3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.
	
4. Clone the repository 

### Environment setup

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
	- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
	- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
	- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

4. Open the `Test_the_environment.ipynb` and run the cells with SHIFT+ENTER. If the environment is correctly installed, you should get to see the Unity environment in another window and values for state and action spaces under `2. Examine the State and Action Spaces`. 


### Train the agent
Open the `p3-Tennis.ipynb` and run the cells with SHIFT+ENTER. 

At the end, weights will be saved both for the critic and actor networks.

