import numpy as np
from torch import nn

def weight_init(x):
    if isinstance(x, nn.Linear):
        nn.init.xavier_normal_(x.weight)
        if x.bias is not None: nn.init.constant_(x.bias, 0.0)
    elif isinstance(x, nn.BatchNorm1d):
        nn.init.constant_(x.weight, 1.0)
        nn.init.constant_(x.bias, 0.0)
        
def scale_action(action, high, low):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    result_action = action * weight + bias

    return result_action

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        self.size = int(max_size)
        self.batch_size = batch_size
        self.count = 0

        self.state_memory = np.zeros((self.size, state_dim))
        self.action_memory = np.zeros((self.size, action_dim))
        self.reward_memory = np.zeros((self.size, ))
        self.next_state_memory = np.zeros((self.size, state_dim))
        self.done_memory = np.zeros((self.size, ), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.count % self.size

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.done_memory[mem_idx] = done

        self.count += 1

    def sample_buffer(self):
        mem_len = min(self.size, self.count)
        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.done_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.count >= self.batch_size

"""
class ReplayBuffer_simplified:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = int(max_size)
        # self.batch_size = batch_size
        self.count = 0

        self.state_memory = np.zeros((self.max_size, state_dim))
        self.action_memory = np.zeros((self.max_size, action_dim))
        self.reward_memory = np.zeros((self.max_size, ))
        self.next_state_memory = np.zeros((self.max_size, state_dim))
        self.done_memory = np.zeros((self.max_size, ), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.count % self.max_size

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.done_memory[mem_idx] = done

        self.count += 1

    def sample_buffer(self, batch_size):
        mem_len = min(self.max_size, self.count)
        batch = np.random.choice(mem_len, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.done_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.count >= self.batch_size
"""