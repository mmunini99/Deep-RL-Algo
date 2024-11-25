import random
from collections import deque
import torch


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
    






