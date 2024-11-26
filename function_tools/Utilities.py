import random
from collections import deque
import torch


def loss_value_PPO(value_new, value_old, rew_to_go, clip_value):

    loss_new = (value_new-rew_to_go)**2
    diff_update = value_old-torch.clamp(value_new-value_old,- clip_value, clip_value)
    loss_old = (diff_update-rew_to_go)**2

    return 0.5*torch.mean(torch.max(loss_new, loss_old))

def loss_policy_PPO(A_value, ratio_is, entropy, clip_value, coef_h):

    first_element = ratio_is*A_value

    second_element = torch.clamp(ratio_is, 1 - clip_value, 1 + clip_value)

    return -torch.mean(torch.min(first_element, second_element)) + coef_h*entropy


# CNN preprocessing
def preprocessing_input_state(state, device):
    state = torch.tensor(state, dtype = torch.float32, device=device)
    state = state.permute(2,0,1).unsqueeze(0)
    return(state)
# IQN
def overall_loss(array_quantiles, array_targets):
    loss = torch.stack([quantile_loss(array_quantiles[:, i], array_targets[:, j]) for i in range(array_quantiles.size(1)) for j in range(array_targets.size(1))])
    return loss.sum()/array_targets.size(1)

# QR- DQN
def quantile_loss(quantiles, targets):
    error = targets - quantiles
    huber_loss = torch.where(torch.abs(error) <= 1, 0.5 * error ** 2, torch.abs(error) - 0.5)
    return huber_loss.mean()

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.store = Transition

    def push(self, *args):
        self.memory.append(self.store(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    






