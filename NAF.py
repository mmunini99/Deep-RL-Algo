import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from itertools import count
from collections import namedtuple
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

from function_tools.Utilities import ReplayMemory, preprocessing_input_state, plot_episode
from function_tools.Environment import CreateEnvironmentContinuous
from function_tools.NNetworks import NAF_DQN



class NAF_Agent():

    def __init__(self, ENV_NAME, BATCH_SIZE, GAMMA, EPSILON, EPSILON_DECAY, STEPS_DECAY, TAU, LR, REPETITION, N_EPISODES, PRINT_PLOT):
        # hyperparameters
        self.env_name = ENV_NAME
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.eps = EPSILON
        self.eps_decay = EPSILON_DECAY
        self.steps_decay = STEPS_DECAY
        self.tau = TAU
        self.lr = LR
        self.repetition = REPETITION
        self.num_episodes = N_EPISODES
        self.print_plot = PRINT_PLOT
        # setting possible accelerator
        self.device = torch.device(
                                    "cuda" if torch.cuda.is_available() else
                                    "cpu"
                                  )
        # build the environemnt for simulations
        self.env = CreateEnvironmentContinuous(self.env_name, self.repetition)
        # get info about the environment
        self.dim_actions = self.env.action_space.shape[0]
        self.action_minimum = torch.from_numpy(self.env.action_space.low).to(self.device)
        self.action_maximum = torch.from_numpy(self.env.action_space.high).to(self.device)
        # define the policy and ist target NN
        self.policy_net = NAF_DQN(self.dim_actions, self.action_maximum, self.device).to(self.device)
        self.target_net = NAF_DQN(self.dim_actions, self.action_maximum, self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        # build the rollout memory
        self.tuple_store = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.memory = ReplayMemory(10000, self.tuple_store)
        # general counter
        self.steps_done = 0
        self.epsiode_durations = []
        self.epsiode_rewards = []

    def noisy_policy(self, state):
        mu = self.policy_net.mean_action(state)
        mu = mu + torch.normal(0, self.eps, mu.size(), device=self.device)
        action = mu.clamp(self.action_minimum, self.action_maximum)
        return action

    def select_action(self, state):
        self.steps_done += 1
        if self.steps_done > 0 and self.steps_done % self.steps_decay == 0:
            if self.eps - self.eps_decay/self.steps_done > 0:
                self.eps -= self.eps_decay/self.steps_done
        return self.noisy_policy(state) # action taken by perturbed state-action value function

    def optimize_model(self):

        if len(self.memory) < self.batch_size: # if not enough data for a training, stop the optimize process
            return

        transitions = self.memory.sample(self.batch_size) # sample data from the rollour store

        batch = self.tuple_store(*zip(*transitions)) # Transpose operation to make it easier to work with individual compnents rather than tuple like : (state, action, reward, next_state, done)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool) # Boolean mask for mapping where the terminal states are
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # prepare the array for state for Q-learning update rule, without the positions of terminal states
                                                    
        state_batch = torch.cat(batch.state)
        state_batch = state_batch.to(self.device)
        action_batch = torch.cat(batch.action)
        action_batch = action_batch.to(self.device)
        reward_batch = torch.cat(batch.reward)
        reward_batch = reward_batch.to(self.device)

        state_action_values = self.policy_net(state_batch, action_batch) # Action taken for each state according to teh trick V(s) + A(s,a), with a the actions took in previous steps and store in the rollout store

        next_state_values = torch.zeros(self.batch_size, device=self.device) # define the array for storing Q(s', a)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net.value_state(non_final_next_states).squeeze(1)
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch # Compute the expected Q values plus the reward. To speed up compuation steps

        # Compute MSE loss 
        criterion = nn.MSELoss() 
        loss = criterion(state_action_values.squeeze(1), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stabilize the training --> Be careful, in-place operations
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def training(self):

        for i_episode in range(self.num_episodes):
            # reward tracker only for plot porpouse
            track_rew = 0
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = preprocessing_input_state(state, self.device) # this allows the correct shape for the CNN
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.cpu().detach().numpy().squeeze(0)) # transform the output of the NN to a format that is accepted by the environment
                track_rew += (reward-track_rew)/(t+1)
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated or truncated:
                    next_state = None
                else:
                    next_state = preprocessing_input_state(observation, self.device) # save directly the correct shape for the next state for CNN, as the previous correction of format won't be repeated

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # Polyak Average
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                

                if done:
                    self.epsiode_durations.append(i_episode)
                    self.epsiode_rewards.append(track_rew)
                    if self.print_plot:
                        plot_episode(self.epsiode_durations, self.epsiode_rewards)
                    break



    def return_metric(self, n_last_episode):
        # here it is called the metric used for hyperparameter tuning
        return np.mean(self.epsiode_rewards[-n_last_episode:])
    

    def return_weights(self):
        # create a dictionary to save the policy net weights and biases (NN) and the optimizer settings
        dict_state = {
                        'policy_dict': self.policy_net.state_dict(),
                        'optimizer_policy_dict': self.optimizer.state_dict()
                      }
        
        return dict_state
    

    def running(self, loading_data):
        # storing
        test_rew = []
        test_steps = []
        # load params
        self.policy_net.load_state_dict(loading_data)
        # running a single episode
        state, info = self.env.reset()
        state = preprocessing_input_state(state, self.device) # this allows the correct shape for the CNN
        for t in count():
            action = self.select_action(state)
            observation, reward, terminated, truncated, _ = self.env.step(action.cpu().detach().numpy().squeeze(0))
            test_rew.append(reward)
            test_steps.append(t)
            reward = torch.tensor([reward], device=self.device)
            done = terminated or truncated

            if terminated or truncated:
                next_state = None
            else:
                next_state = preprocessing_input_state(observation, self.device) # save directly the correct shape for the next state for CNN, as the previous correction of format won't be repeated

            # Move to the next state
            state = next_state

            if done:
                break

        return test_steps, test_rew



            




