# DeepRL
This repository is dedicated to the implementation of deep reinforcement learning algorithms, implemented in Python3. The neural network framework used to train models is Torch (PyTorch).

The source code provides the implementation of elements described in the DQN paper. 

### Dependencies
Install Pytorch:
Instructions on https://pytorch.org/

Install Unity ml-agent library:
https://github.com/Unity-Technologies/ml-agents

Environment file located in:  
`data/Banana_Linux_NoVis/Banana.x86_64`


### Environment details
The objective of the project is to train an agent to navigate in the "Banana" environment in order to catch only yellow bananas as fast as possible. The goal of the agent is to learn the Q-function related to 37 continuous state variables returned by the environment, and 4 navigation actions (left, right, forward bacward). After each action, the reward is 0 if the agent does not catch any banana, +1 if it catches a yellow banana and -1 for a blue banana. An episode ends after 1000 steps.

### Launch training
Code is located in `dqn/`.  
Setup the hyperparameters in:
* `main.py`: exploration parameters (epsilon related coefficients)
* `model.py`: model architecture (layers size)
* `agent.py`: training parameters

To launch training:  
`python main.py`

After training, trained model (weights), are saved to `ckpt/checkpoint.pth`
