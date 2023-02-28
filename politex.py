import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# !pip install -U gym
from scipy.special import softmax
import csv


def Discrete(obs, bins):
    obs_index = []
    for i in range(len(obs)): 
        obs_index.append(np.digitize(obs[i], bins[i]) - 1)
    return tuple(obs_index)

# Set hyperparameters
gamma = 0.995  # Discount factor
alpha = 0.001  # Learning rate
eta = 1.
# tau = 1000 # Phase length
max_episodes = int(1e4)  # Maximum number of iterations
epsilon = 0.
batch_size = int(1024) 
print_every = 100

# Create the environments
env_name = "CartPole-v1"
# env_name = "FrozenLake-v1"
env = gym.make(env_name, render_mode="human")     #, is_slippery=False)#, render_mode="human")

action_dim =  env.action_space.n # number of possible actions
if env_name == "CartPole-v1":
  state_dim =  env.observation_space.shape[0] # dimension of state space
  bin_size=30
  bins = [np.linspace(-4.8,4.8,bin_size),
            np.linspace(-4,4,bin_size),
            np.linspace(-0.418,0.418,bin_size),
            np.linspace(-4,4,bin_size)]
  Q = np.zeros(([bin_size] * state_dim + [action_dim]))
elif env_name == "FrozenLake-v1":
  state_dim =  env.observation_space.n # number of states
  Q = np.zeros(([state_dim] + [action_dim]))
else:
   raise ValueError



# Keep track of rewards and run time
rewards = []
av_rewards = []
phase_lengths = []
# Initialize the batch to empty, Q(s,a) and the accumulated reward to zeros
Z = []
# Q, bins = initi_Q(state_dim, action_dim)
Q_hat = Q.copy()
total_reward = 0

# Algorithm: Politex
# Reset the environment and sample an initial state obs
obs, info = env.reset(seed=123, options={})
if env_name == "CartPole-v1":
  obs = Discrete(obs, bins)

for e in tqdm(range(max_episodes)):
  # Initialize policy
  policy = np.array(softmax(eta*Q_hat, axis=-1))
  total_reward = 0
  n_steps = 0

  while True:
    n_steps+=1 
    # Execute current policy
    if np.random.uniform(0,1) < epsilon:
      action = env.action_space.sample() # Random action (left or right).
    else:
      # Softmax policy
      # action = np.random.choice([i for i in range(action_dim)], p=policy[obs])
      # Greedy action 
      qs = [Q[obs][a] for a in range(action_dim)]
      action = np.argmax(qs) 
      

    # Take action a and observe the reward and next state
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if env_name == "CartPole-v1":
      next_obs = Discrete(next_obs, bins)

    # Store experience in replay buffer
    Z.append((obs, action, reward, next_obs, done))

    # update current obs and total reward
    obs = next_obs
    total_reward += reward

    if done:
        obs, info = env.reset()
        if env_name == "CartPole-v1":
          obs = Discrete(obs, bins)
        break


  # Sample a random batch of experiences from Z
  # batch_size = int(len(Z)*.8)
  # batch = random.sample(Z, batch_size)
  batch = random.sample(Z, min(batch_size, len(Z)))

  list_td_erros = []
  # Compute Q-values using data in Z
  for jj in range(100): # TODO we have to implement more steps here to decrease the TD error (even for same sample) 
    for obs_batch, action_batch, reward_batch, next_obs_batch, done_batch in batch:
      Q_next_obs = np.array([Q[next_obs_batch][a] for a in range(env.action_space.n)])
      # max_Q_next_obs = max(Q_next_obs)
      # V_next_obs = max(Q_next_obs) 
      V_next_obs = Q_next_obs @ policy[next_obs]
      # Calculate TD error
      td_error = reward_batch + gamma * (1-terminated)*V_next_obs - Q[obs_batch][action_batch]
      list_td_erros.append(td_error)
      # Update Q-value for current state-action pair
      Q[obs_batch][action_batch] += alpha * td_error
    

  # Q_hat = e * Q_hat / (e+1) + Q / (e+1) # TODO do we really want to average? Is like decreasing eta?
  Q_hat +=  Q 


  # Print progress every 100 iterations
  rewards.append(total_reward)
  if (e + 1) % print_every == 0:
    av_rewards = np.mean(np.array(rewards[-100:]))
    print(f'Episode {e + 1}/{max_episodes} | Last total reward: {total_reward} | '
          f'Average reward in last 100 episodes: {av_rewards} | Last episode length: {n_steps}')