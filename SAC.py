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
from function_tools.NNetworks import POLICY__SAC, Q__SAC



class SAC_Agent():

    def __init__(self, ENV_NAME, BATCH_SIZE, GAMMA, ENTROPY_PARAM, K_EPOCHS, STEPS_UPDATE, TAU, LR, REPETITION, N_EPISODES, PRINT_PLOT):
        # hyperparameters
        self.env_name = ENV_NAME
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.entropy_param = ENTROPY_PARAM
        self.k_epochs = K_EPOCHS
        self.steps_update = STEPS_UPDATE
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
        self.policy_net = POLICY__SAC(self.dim_actions, self.action_maximum).to(self.device)
        self.target_policy_net = POLICY__SAC(self.dim_actions, self.action_maximum).to(self.device)
        self.Q1_net = Q__SAC(self.dim_actions, self.device).to(self.device)
        self.Q2_net = Q__SAC(self.dim_actions, self.device).to(self.device)
        self.target_Q1 = Q__SAC(self.dim_actions, self.device).to(self.device)
        self.target_Q2 = Q__SAC(self.dim_actions, self.device).to(self.device)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_Q1.load_state_dict(self.Q1_net.state_dict())
        self.target_Q2.load_state_dict(self.Q2_net.state_dict())

        self.optimizer_Q1 = optim.AdamW(self.Q1_net.parameters(), lr=LR, amsgrad=True)
        self.optimizer_Q2 = optim.AdamW(self.Q2_net.parameters(), lr=LR, amsgrad=True)
        self.optimizer_policy = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        # build the rollout memory
        self.tuple_store = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.memory = ReplayMemory(10000, self.tuple_store)
        # general counter
        self.steps_done = 0
        self.epsiode_durations = []
        self.epsiode_rewards = []

    def select_action(self, state):
        self.steps_done += 1
        with torch.no_grad():

            action, _ = self.policy_net(state)

            return action 

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

        ### Q - FUNCTION LOSS

        state_action_values1 = self.Q1_net(state_batch, action_batch)
        state_action_values2 = self.Q2_net(state_batch, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        log_action_values= torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            future_action_target, log_prob_action_target = self.target_policy_net(non_final_next_states)
            min_target_Q = torch.min(
                self.target_Q1(non_final_next_states, future_action_target),
                self.target_Q2(non_final_next_states, future_action_target)
            )






            next_state_values[non_final_mask] = min_target_Q.squeeze(1)
            log_action_values[non_final_mask] = log_prob_action_target.squeeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values - self.entropy_param*log_action_values)*self.gamma + reward_batch

        # Compute MSE loss
        criterion = nn.MSELoss()
        loss_Q1 = criterion(state_action_values1, expected_state_action_values.unsqueeze(1))
        loss_Q2 = criterion(state_action_values2, expected_state_action_values.unsqueeze(1))



        ### POLICY-FUNCTION LOSS
        action_loss, log_prob_action_loss = self.policy_net(state_batch)

        Q_min_value = torch.min(self.Q1_net(state_batch, action_loss), self.Q2_net(state_batch, action_loss))

        loss_policy = (Q_min_value - self.entropy_param*log_prob_action_loss).mean()


        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer_policy.step()

        # Optimize the model
        self.optimizer_Q1.zero_grad()
        loss_Q1.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.Q1_net.parameters(), 100)
        self.optimizer_Q1.step()


        self.optimizer_Q2.zero_grad()
        loss_Q2.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.Q2_net.parameters(), 100)
        self.optimizer_Q2.step()

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
                if self.steps_done > 0 and self.steps_done % self.steps_update:
                        # Perform one step of the optimization (on both NN)
                        for _ in range(self.k_epochs):
                            self.optimize_model()

                        # Soft update of the target network's weights
                        # Polyak Average
                        target_Q1_state_dict = self.target_Q1.state_dict()
                        target_Q2_state_dict = self.target_Q2.state_dict()
                        target_policy_net_state_dict = self.target_policy_net.state_dict()

                        Q1_net_state_dict = self.Q1_net.state_dict()
                        Q2_net_state_dict = self.Q2_net.state_dict()
                        policy_net_state_dict = self.policy_net.state_dict()


                        for keyQ1 in Q1_net_state_dict:
                            target_Q1_state_dict[keyQ1] = Q1_net_state_dict[keyQ1]*self.tau + target_Q1_state_dict[keyQ1]*(1-self.tau)
                        self.target_Q1.load_state_dict(target_Q1_state_dict)
                        for keyQ2 in Q2_net_state_dict:
                            target_Q2_state_dict[keyQ2] = Q2_net_state_dict[keyQ2]*self.tau + target_Q2_state_dict[keyQ2]*(1-self.tau)
                        self.target_Q2.load_state_dict(target_Q2_state_dict)
                        for key in policy_net_state_dict:
                            target_policy_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_policy_net_state_dict[key]*(1-self.tau)
                        self.target_policy_net.load_state_dict(target_policy_net_state_dict)

                

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
        # create a dictionary to save the policy and Q(s,a) nets weights and biases (NN) and the optimizers settings
        dict_state = {
                        'policy_dict': self.policy_net.state_dict(),
                        'q1_dict': self.Q1_net.state_dict(),
                        'q2_dict': self.Q2_net.state_dict(),
                        'optimizer_policy_dict': self.optimizer_policy.state_dict(),
                        'optimizer_q1_dict': self.optimizer_Q1.state_dict(),
                        'optimizer_q2_dict': self.optimizer_Q2.state_dict(),
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
            




