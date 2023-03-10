import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# !pip install -U gym
from scipy.special import softmax
import csv


def initi_Q(state_space, action_space, bin_size=30):
    
    bins = [np.linspace(-4.8,4.8,bin_size),
            np.linspace(-4,4,bin_size),
            np.linspace(-0.418,0.418,bin_size),
            np.linspace(-4,4,bin_size)]
    
    q_table = np.zeros(([bin_size] * state_space + [action_space]))
    return q_table, bins

def Discrete(obs, bins):
    obs_index = []
    for i in range(len(obs)): 
        obs_index.append(np.digitize(obs[i], bins[i]) - 1)
    return tuple(obs_index)

# Set hyperparameters
gamma = 0.995  # Discount factor
alpha = 0.1  # Learning rate
eta = 10.
tau = 1000 # Phase length
max_iterations = 1000  # Maximum number of iterations
epsilon = -0.01
batch_size = int(1024) 
print_every = 100

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Keep track of rewards and run time
rewards = []
av_rewards = []
episodes_lengths = []

state_dim =  env.observation_space.shape[0] # dimension of state space
action_dim =  env.action_space.n # number of possible actions

# Initialize the batch to empty, Q(s,a) and the accumulated reward to zeros
Z = []
Q, bins = initi_Q(state_dim, action_dim)
Q_hat = Q.copy()
total_reward = 0

# Algorithm: Politex
# Reset the environment and sample an initial state obs
obs, info = env.reset()
obs = Discrete(obs, bins)

for i in tqdm(range(max_iterations)):

  # Initialize policy
  policy = np.array(softmax(eta*Q_hat, axis=-1))
  total_reward = 0
  episode_length = 0

  for t in range(tau):
    episode_length +=1
    # Execute current policy
    if np.random.uniform(0,1) < epsilon:
      action = env.action_space.sample() # Random action (left or right).
    else:
      action = np.random.choice([0, 1], p=policy[obs])

      # qs = [Q[Discrete(s, bins)][a] for a in range(env.action_space.n)]
      # a = np.argmax(qs) # Greedy action for state.

    # Take action a and observe the reward and next state s'
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    next_obs = Discrete(next_obs, bins)

    # Store experience in replay buffer
    Z.append((obs, action, reward, next_obs, done))

    # Set current state to next state
    obs = next_obs

    total_reward += reward

    if done:
      obs, info = env.reset()
      obs = Discrete(obs, bins)
      # Store episode reward and episode length
      episodes_lengths.append(episode_length) # TODO this is basically the total_rewards-1 since reward is +1 for each time step, do we need it?
      rewards.append(total_reward)
      # reinitilize total reward for next episode
      total_reward = 0
      episode_length = 0 
      # break

  # eta -= 0.0005 # TODO why do we do this?

  # Sample a random batch of experiences from Z
  # batch_size = int(len(Z)*.8) # TODO why do we do this? In general you have a parameter that fix the length of the batch size (I changed it below)
  # batch = random.sample(Z, batch_size)
  batch = random.sample(Z, min(batch_size, len(Z)))

  # Compute Q-values using data in Z  
  # TODO 1) Here the problem was that alpha was too small. 
  # We should also think that making more steps with the same tuple (s,a,r...) can be beneficial, so we might rethink the loop below. 
  # I am not sure anyway how to change it, we need to try. 
  # But for sure if alpha is too small and you dont do more steps on the same state-actions then Q changes are too small for each (s,a) pair.
  # TODO 2) before you were using an argmax over Q wich is good for Q learning but here we just try to estimate Q_\pi for 
  # policy \pi. Therefore the idea is to compute TD
  for obs_batch, action_batch, reward_batch, next_obs_batch, done_batch in batch:
    Q_next_obs = [Q[next_obs_batch][a] for a in range(env.action_space.n)]
    # V_next_obs = max(Q_next_obs) # TODO this should be theoretically wrong but works well in practice 
    V_next_obs = Q_next_obs @ policy[next_obs]
    # Calculate TD error
    td_error = reward_batch + gamma * V_next_obs - Q[obs_batch][action_batch]
    # Update Q-value for current state-action pair
    Q[obs_batch][action_batch] += alpha * td_error

  Q_hat = i * Q_hat / (i+1) + Q / (i+1) # TODO check this is the right fomula for cumulative average

  
  
  
  # if i < 950: # TODO I don't get this
  #   epsilon -= 0.00101

  # Print progress every 100 iterations
  rewards.append(total_reward)
  if (i + 1) % print_every == 0:
    av_rewards = np.mean(np.array(rewards[-100:]))
    print(f'Iteration {i + 1}/{max_iterations} | Last total reward: {total_reward} | '
          f'Average reward in last 100 episodes: {av_rewards} | Last episode length: {episodes_lengths[-1]}')
