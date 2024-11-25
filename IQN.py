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
from concurrent.futures import ThreadPoolExecutor

from function_tools.Utilities import ReplayMemory, overall_loss
from function_tools.Environment import CreateEnvironment
from function_tools.NNetworks import IQN



class IQN_Agent():

    def __init__(self, ENV_NAME, BATCH_SIZE, GAMMA, EPS_START, EPS_DECAY, EPS_END, TAU, LR, SUB_AGENTS, N_EPISODES):
        # hyperparameters
        self.env_name = ENV_NAME
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.eps_end = EPS_END
        self.tau = TAU
        self.lr = LR
        self.n_sub_agents = SUB_AGENTS
        self.num_episodes = N_EPISODES
        # setting possible accelerator
        self.device = torch.device(
                                    "cuda" if torch.cuda.is_available() else
                                    "cpu"
                                  )
        # build the environemnt for simulations
        self.env = CreateEnvironment(self.env_name)
        # get info about the environment
        self.n_actions = self.env.action_space.n
        self.n_observations = self.env.observation_space.shape[0]
        # define the policy and ist target NN
        self.policy_net = IQN(self.n_observations, self.n_actions, self.device).to(self.device)
        self.target_net = IQN(self.n_observations, self.n_actions, self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        # build the rollout memory
        self.tuple_store = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.memory = ReplayMemory(10000, self.tuple_store)
        # general counter
        self.steps_done = 0
        self.epsiode_durations = []
        self.epsiode_rewards = []


    def select_action(self, state):

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay) # time-decay of the exploration tentatives

        self.steps_done += 1
        if sample > eps_threshold: # standard eps-greedy policy
            with torch.no_grad():
                Q_quantiles_net = self.policy_net(state) # compute Q(s,a)
                return torch.tensor([[torch.argmax(Q_quantiles_net).item()]], device=self.device, dtype=torch.long) # choose action with the greatest state-action value for exploitation purpose
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long) # chose the action randomly for exploration purpose
        

    def parallel_forward(self, input_tensors):

        state_batch, action_batch, non_final_mask, non_final_next_states, reward_batch = input_tensors
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        non_final_mask = non_final_mask.to(self.device)
        non_final_next_states = non_final_next_states.to(self.device)
        reward_batch = reward_batch.to(self.device)

        state_action_values = self.policy_net(state_batch) # compute Q(s,a)
        state_action_values = torch.gather(state_action_values, 1, action_batch) # Action taken for each state according to Q(s,a), with a the actions took in previous steps and store in the rollout store
        state_action_values = state_action_values.squeeze(-1) 



        next_state_values = torch.zeros(self.batch_size, device=self.device) # define the array for storing Q(s', a)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values


        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch # Compute the expected Q values plus the reward. To speed up compuation steps


        return state_action_values, expected_state_action_values

    def run_parallel_forward(self, input_tensors):
        return self.parallel_forward(input_tensors)

    def optimize_model(self):

        if len(self.memory) < self.batch_size: # if not enough data for a training, stop the optimize process
            return

        transitions = self.memory.sample(self.batch_size) # sample data from the rollour store
        batch = self.tuple_store(*zip(*transitions)) # Transpose operation to make it easier to work with individual compnents rather than tuple like : (state, action, reward, next_state, done)

        all_state_action_values = [] 
        all_expected_state_action_values = []

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool) # Boolean mask for mapping where the terminal states are
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # prepare the array for state for Q-learning update rule, without the positions of terminal states
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)



        with ThreadPoolExecutor() as executor: # parallelization of sub-agents
            input_tensors = [(state_batch, action_batch, non_final_mask, non_final_next_states, reward_batch)]*self.n_sub_agents
            results = list(executor.map(self.run_parallel_forward, input_tensors))



        for state_action_values, expected_state_action_values in results: # unify the differemt experiences
            all_state_action_values.append(state_action_values)
            all_expected_state_action_values.append(expected_state_action_values)



        all_state_action_values = torch.stack(all_state_action_values).transpose(0,1)
        all_expected_state_action_values = torch.stack(all_expected_state_action_values).transpose(0,1)

        # Compute loss for different experiences based on the Huber loss for quantiles
        loss = overall_loss(all_state_action_values, all_expected_state_action_values)


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def training(self):

        for i_episode in range(self.num_episodes):
            # reward tracker only for plot porpouse
            track_rew = 0
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) # this allows the correct shape for the MLP
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                track_rew += (reward-track_rew)/(t+1)
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated or truncated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0) # save directly the correct shape for the next state for MLP, as the previous correction of format won't be repeated

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                

                if done:
                    self.epsiode_durations.append(i_episode)
                    self.epsiode_rewards.append(track_rew)
                    self.plot_episode()
                    break


    def plot_episode(self):

        is_ipython = 'inline' in matplotlib.get_backend()         

        plt.ion()

        plt.figure(1)

        plt.clf()
        plt.title('Training...')  

        plt.xlabel('Episode')
        plt.ylabel('Mean rewards') 

        array_epis = np.array(self.epsiode_durations)  
        array_rew = np.array(self.epsiode_rewards)  

        plt.plot(array_epis, array_rew)

        plt.pause(0.001) 

        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)



            




