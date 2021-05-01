import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "./python"])

from collections import deque
import numpy as np
import torch
from unityagents import UnityEnvironment
from agent import Agent
from matplotlib import pylab as plt
"""
This script allows to train an agent using the DQN algorithm.
To launch the script: 'python main.py'.
"""

def dqn_training(env, agent, brain_name="BananaBrain", n_episodes=1000,
                 max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    This function launch the agent training.

    It initializes:
        - variables to track the score evolution over episodes.
        - epsilon value for epsilon greedy

    It allows to loop over episodes until the training ending condition
    (avg(scores_window)>=13), and to update the dqn with the training algorithm
    ('step' method of the 'agent' object defined in the Agent module).

    At the end of the learning, the trained model is stored on disk.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy
        action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing
        epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset env
        state = env_info.vector_observations[0]            # get current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]  # send the action to env
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: \
                  {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


def main():
    """
    Script which launch the agent learning.
    It creates the agent and environment instances, and launch the agent
    training.
    """
    # Load environment
    env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    # Initialize the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of states
    state = env_info.vector_observations[0]
    state_size = len(state)

    # Instantiate an agent
    agent = Agent(state_size, action_size, seed=1)

    # Launch agent training
    scores = dqn_training(env, agent)

    # plot the scores
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == "__main__":
    main()

