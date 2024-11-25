import random
from collections import deque

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.store = Transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.store(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    



