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

from function_tools.Utilities import ReplayMemory, preprocessing_input_state, loss_value_PPO, loss_policy_PPO, plot_episode
from function_tools.Environment import CreateEnvironmentContinuous
from function_tools.NNetworks import POLICY__PPO, VALUE__PPO



class PPO_Agent():

    def __init__(self, ENV_NAME, BATCH_SIZE, NUM_BATCH_MAX, GAMMA, TRUNC_PARAM, LAMBDA, MAX_LEN_TRAJ, N_ACTORS, K_EPOCHS, CLIP_VALUE, COEF_H, LR, REPETITION, NUM_ITERATIONS, PRINT_PLOT):
        # hyperparameters
        self.env_name = ENV_NAME
        self.batch_size = BATCH_SIZE
        self.num_batch_max = NUM_BATCH_MAX
        self.gamma = GAMMA
        self.trunc_param = TRUNC_PARAM
        self.lambd = LAMBDA
        self.max_len_traj = MAX_LEN_TRAJ
        self.n_actors = N_ACTORS
        self.k_epochs = K_EPOCHS
        self.clip_value = CLIP_VALUE  
        self.coef_h = COEF_H
        self.lr = LR
        self.repetition = REPETITION
        self.num_iterations = NUM_ITERATIONS
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
        self.action_maximum = torch.from_numpy(self.env.action_space.high).to(self.device)
        # define the policy and ist target NN
        self.policy_net = POLICY__PPO(self.dim_actions, self.action_maximum).to(self.device)
        self.value_net = VALUE__PPO().to(self.device)
        self.optimizer_policy = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.optimizer_value = optim.AdamW(self.value_net.parameters(), lr=self.lr, amsgrad=True)
        # general counter
        self.steps_done = 0
        self.track_episode = 0
        self.epsiode_durations = []
        self.epsiode_rewards = []

    def select_action(self, state):
        with torch.no_grad():

            action, log_prob = self.policy_net(state) # for each action, defined in a continuous way, it is computed the associated stabilized log-probability from Normal pdf

        return action, log_prob
    
    def compute_value(self, state):
        with torch.no_grad():

            value = self.value_net(state)

        return value
    

    def return_to_go(self, tensor_rewards):

        weights = self.gamma ** torch.arange(tensor_rewards.size(0))


        weights = weights.to(self.device)

        # Compute weighted rewards
        weighted_rewards = tensor_rewards * weights


        # Compute cumulative sum of weighted rewards
        return torch.cumsum(weighted_rewards.flip(0), dim=0).flip(0)
    

    def compute_TD_error(self, value_current, value_next, reward, status_done):

        if status_done == True:

            return reward - value_current

        else:

            return reward + self.gamma*value_next - value_current


    def compute_tgae(self, store_value, store_rewards, store_end):

        T = store_rewards.size(0)#len of the trajectory
        # Initialize advantage array
        advantages = np.zeros(T)

        # Compute T-GAE for each time step
        for t in range(T-1):
            adv = self.compute_TD_error(store_value[t], store_value[t+1], store_rewards[t], store_end[t])

            # Apply truncation (sum over a fixed number of steps)
            for k in range(1, min(self.trunc_param, T - 1 - t)):
                adv += (self.gamma * self.lambd)**k * self.compute_TD_error(store_value[t+k], store_value[t+k+1], store_rewards[t+k], store_end[t+k])
            advantages[t] = adv

        return torch.tensor((advantages - advantages.mean()) / (advantages.std() + 0.000001)) # 1e-5 for stabilizing and assure continuity to the standardization




    def collect_set_trajectory(self):
        store_state = []
        store_action = []
        store_log_probability_pi = []
        store_rewards = []
        store_len_episode = []
        store_end = []
        store_value = []

        track_rew = 0 # reward tracker only for plot porpouse
        track_count_step = 0

        timestep_episode = 0 #counter for closing update session


        state, _ = self.env.reset() #reset the environemnt each time an episode ends


        while timestep_episode < self.max_len_traj: #collect data for update

            store_state.append(preprocessing_input_state(state, self.device)) #store the state in the correct shape for the CNN

            action_taken, log_probability_pi = self.select_action(preprocessing_input_state(state, self.device)) #run the old policy

            value_taken = self.compute_value(preprocessing_input_state(state, self.device))

            store_value.append(value_taken)


            next_state, reward, done, truncated, _ = self.env.step(action_taken.cpu().detach().numpy().squeeze(0)) # transform the output of the NN to a format that is accepted by the environment

            track_rew += (reward-track_rew)/(track_count_step+1)

            store_rewards.append(reward) #store the reward
            store_log_probability_pi.append(log_probability_pi.tolist()[0]) #store log prob of the action --> for Importance Sampling
            store_action.append(action_taken.tolist()[0])
            store_len_episode.append(timestep_episode)


            timestep_episode += 1

            if done or truncated:
                store_end.append(True)
                self.track_episode += 1
                self.epsiode_durations.append(self.track_episode)
                self.epsiode_rewards.append(track_rew)
                if self.print_plot:
                    plot_episode(self.epsiode_durations, self.epsiode_rewards)
                track_count_step = 0
                track_rew = 0
                state, _ = self.env.reset()
                
                

            else:
                track_count_step += 1
                state = next_state
                store_end.append(False)

        store_state = torch.stack(store_state).squeeze(1).to(self.device)
        store_action = torch.tensor(store_action, dtype = torch.float, device = self.device)
        store_log_probability_pi = torch.tensor(store_log_probability_pi, dtype = torch.float, device = self.device)
        store_rewards  = torch.tensor(store_rewards, dtype = torch.float, device = self.device)
        store_end = torch.tensor(store_end, dtype = torch.bool, device = self.device)
        store_value = torch.tensor(store_value, dtype = torch.float, device = self.device)




        return store_state, store_action.squeeze(1), store_log_probability_pi.squeeze(1), store_rewards, store_len_episode, store_end, store_value


    def compute_dataset(self):
        batch_state = []
        batch_action = []
        batch_log_probability_pi = []
        batch_rewards_to_go = []
        batch_A_value = []
        batch_end = []
        batch_value = []





        for _ in range(self.n_actors):

            store_state, store_action, store_log_probability_pi, store_rewards, store_len_episode, store_end, store_value = self.collect_set_trajectory()
            batch_state.append(store_state)
            batch_action.append(store_action)
            batch_log_probability_pi.append(store_log_probability_pi)
            batch_rewards_to_go.append(self.return_to_go(store_rewards))
            batch_A_value.append(self.compute_tgae(store_value, store_rewards, store_end))
            batch_end.append(store_end)
            batch_value.append(store_value)


        batch_state = torch.cat(batch_state).to(self.device)
        batch_rewards_to_go = torch.cat(batch_rewards_to_go).to(self.device)
        batch_A_value = torch.cat(batch_A_value).to(self.device)
        batch_action = torch.cat(batch_action)
        batch_log_probability_pi = torch.cat(batch_log_probability_pi).to(self.device)
        batch_end = torch.cat(batch_end).to(self.device)
        batch_value = torch.cat(batch_value).to(self.device)


        return batch_state, batch_action, batch_log_probability_pi, batch_rewards_to_go, batch_A_value, batch_end, batch_value


    def batch_index(self, len_max):
        # Create a range of indices from 0 to len_max - 1
        indices = torch.arange(0, len_max)

        # Shuffle the indices randomly
        shuffled_indices = torch.randperm(len_max)

        # Split the shuffled indices into batches
        sub_index_tensors = [shuffled_indices[i:i + self.batch_size] for i in range(0, len_max, self.batch_size)]

        return sub_index_tensors

    def training(self):

        while self.steps_done < self.num_iterations:

            data_state, data_action, data_log_pi, data_ret_to_go, data_adv, data_end, data_value = self.compute_dataset()


            for ep in range(self.k_epochs):

                indices = self.batch_index(data_state.size(0))

                for id in range(0, self.num_batch_max):

                    idx = indices[id]
                    idx = idx.to(self.device)

                    state_batch = data_state[idx]
                    action_batch = data_action[idx]
                    logpi_batch = data_log_pi[idx]
                    ret_batch = data_ret_to_go[idx]
                    adv_batch = data_adv[idx]
                    end_batch = data_end[idx]
                    value_batch = data_value[idx]

                    state_batch = state_batch.to(self.device)
                    action_batch = action_batch.to(self.device)
                    logpi_batch = logpi_batch.to(self.device)
                    ret_batch = ret_batch.to(self.device)
                    adv_batch = adv_batch.to(self.device)
                    end_batch = end_batch.to(self.device)
                    value_batch = value_batch.to(self.device)


                    #define new value
                    new_value = self.value_net(state_batch)
                    new_value = new_value.squeeze(1)
                    new_value = new_value.to(self.device)


                    #define new log pi and entropy
                    entropy, new_log_pi = self.policy_net.log_prob_and_entropy(state_batch, action_batch)


                    #ratio of prob
                    ratio_imp_sampling = torch.exp(new_log_pi - logpi_batch)
                    ratio_imp_sampling = ratio_imp_sampling.to(self.device)

                    #loss policy
                    loss_pi = loss_policy_PPO(adv_batch, ratio_imp_sampling, entropy, self.clip_value, self.coef_h)

                    #loss value
                    loss_v = loss_value_PPO(new_value, value_batch, ret_batch, self.clip_value)

                    self.optimizer_policy.zero_grad()
                    loss_pi.backward()
                    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
                    self.optimizer_policy.step()


                    self.optimizer_value.zero_grad()
                    loss_v.backward()
                    torch.nn.utils.clip_grad_value_(self.value_net.parameters(), 100)
                    self.optimizer_value.step()

            self.steps_done += 1








            




