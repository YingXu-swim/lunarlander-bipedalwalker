import sys, os
sys.path.insert(0, os.path.abspath(".."))
import copy
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from common import helper as h
# from common.buffer import ReplayBuffer # 
from utils import ReplayBuffer, weight_init
# from utils import weight_init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def scale_action(action, low, high):
    action = action * (high - low) + low
    return action

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        # add LayerNorm
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 400), nn.LayerNorm(400), nn.ReLU(),
            nn.Linear(400, 300), nn.LayerNorm(300), nn.ReLU(),
            nn.Linear(300, action_dim)
        )
        # initualize
        self.apply(weight_init)
        self.to(device)

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # adjust to spped up training process
        # ----------------------------------
        # for state:
        self.func1 = nn.Linear(state_dim, 400)
        self.norm1 = nn.LayerNorm(400)

        self.func2 = nn.Linear(400, 300)
        self.norm2 = nn.LayerNorm(300)
        # ------------------------------------
        # for action:
        self.func3 = nn.Linear(action_dim, 300)
        # ----------------------------------------
        # for (state + action)
        self.func4 = nn.Linear(300, 1)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.001)
        # initualize
        self.apply(weight_init)
        self.to(device)

    def forward(self, state, action):
        # can run faster 
        # if process state and action individually  then combine  
        # state processing
        state = torch.relu(self.norm1(self.func1(state)))
        state = self.norm2(self.func2(state))
        # action processing
        action = self.func3(action)

        # combination
        feature = torch.relu(state + action)
        return self.func4(feature)

class DDPG:
    def __init__(self, 
                state_shape, 
                action_dim,
                max_action,
                actor_lr,
                critic_lr,
                gamma, 
                tau,
                batch_size,
                use_ou=False,
                normalize=False,
                buffer_size=1e6
                ):
        state_dim = state_shape[0]
        self.action_dim = action_dim
        self.max_action = max_action

        self.pi = Policy(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=actor_lr)

        self.q = Critic(state_dim=state_dim, action_dim=action_dim)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=critic_lr, weight_decay=0.001)
        
        self.buffer = ReplayBuffer(max_size=int(buffer_size), state_dim=state_dim, action_dim=action_dim,batch_size=batch_size)# gai
        # self.buffer = ReplayBuffer(state_shape, action_dim, max_size=int(buffer_size))

        if normalize:
            self.state_scaler = h.StandardScaler(n_dim=state_dim)
        else:
            self.state_scaler = None

        if use_ou:
            self.noise = h.OUActionNoise(mu=np.zeros((action_dim,)))
        else:
            self.noise = None
            
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        # adjust to 0 
        self.random_transition = 0  # collect 5k random data for better exploration

    def update(self):
        
        # if self.buffer.ptr < self.batch_size:
        if not self.buffer.ready():
            return

        # batch = self.buffer.sample(self.batch_size, device=device)
        states, actions, reward, states_, terminals = self.buffer.sample_buffer()

        # in order to solve error: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
        """
        states = torch.tensor(batch.state, dtype=torch.float).to(device)
        action = torch.tensor(batch.action, dtype=torch.float).to(device)
        reward = torch.tensor(batch.reward, dtype=torch.float).to(device)
        next_state = torch.tensor(batch.next_state, dtype=torch.float).to(device)
        not_done = torch.tensor(batch.not_done).to(device)
        """
        states = torch.tensor(states, dtype=torch.float).to(device)
        action = torch.tensor(actions, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        next_state = torch.tensor(states_, dtype=torch.float).to(device)
        done = torch.tensor(terminals).to(device)

        """
        if self.state_scaler is not None:
            self.state_scaler.fit(state)
            states = self.state_scaler.transform(states)
            next_states = self.state_scaler.transform(next_state)
        else:
            states = batcstate
            next_states = batch.next_state
        """

        # compute target q
        with torch.no_grad():
            next_action = (self.pi.forward(next_state)).clamp(-self.max_action, self.max_action)
            q_tar = self.q_target.forward(next_state, next_action).view(-1)
            # td_target = reward + self.gamma * not_done * q_tar
            q_tar[done] = 0.0
            td_target = reward + self.gamma * q_tar

        
       # compute current q
        q_cur = self.q.forward(states, action).view(-1)

        # compute critic loss
        critic_loss = F.mse_loss(q_cur, td_target.detach())

        # optimize the critic
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # compute actor loss
        # actor_loss = -torch.mean(self.q(states_tensor, self.pi.forward(states_tensor)))
        actor_loss = -self.q.forward(states, self.pi.forward(states)).mean()

        # optimize the actor
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        h.soft_update_params(self.q, self.q_target, self.tau)
        h.soft_update_params(self.pi, self.pi, self.tau)
        return {'q': q_cur.mean().item()}
        
    def get_action(self, observation, evaluation=False):
        self.pi.eval()

        if observation.ndim == 1:
            observation = observation[None]  # add the batch dimension

        x = torch.from_numpy(observation).float().to(device)
        # state = torch.tensor([observation], dtype=torch.float).to(device)

        if self.state_scaler is not None:
            x = self.state_scaler.transform(x)

        if self.buffer_ptr < self.random_transition:  # collect random trajectories for better exploration.
            action_ = torch.rand(self.action_dim)
            action = scale_action(action_, -self.max_action, self.max_action)########## edit
        else:
            expl_noise = 0.1  # the stddev of the expl_noise if not evaluation
            ########## Your code starts here. ##########
            # Use the policy to calculate the action to execute
            # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
            # Hint: Make sure the returned action shape is correct.
            # pass

            action = self.pi.forward(x).squeeze()
            # action = self.pi.forward(x)
            if not evaluation: # train
                if self.noise is not None:
                    action = action + torch.from_numpy(self.noise()).float().to(device)
                else:
                    action = action + expl_noise * torch.rand_like(action)
                    action = torch.clamp(action, -self.max_action, self.max_action)
                    
            self.pi.train()
            
        # return action, {}
        return action.detach().cpu().numpy(), {}

    def record(self, state, action, reward, next_state, done):
        # self.buffer_ptr += 1
        # self.buffer.add(state, action, next_state, reward, done)
        self.buffer.store_transition(state, action, reward, next_state, done)

    # You can implement these if needed, following the previous exercises.
    def load(self, filepath, ep, seed):
        self.pi.load_state_dict(torch.load(f'{filepath}/actor_run_test'+str(seed)+str(ep)+'.pt'))
        self.pi_target.load_state_dict(torch.load(f'{filepath}/actor_run_test'+str(seed)+str(ep)+'.pt'))
        self.q.load_state_dict(torch.load(f'{filepath}/critic_run_test'+str(seed)+str(ep)+'.pt'))
        self.q_target.load_state_dict(torch.load(f'{filepath}/critic_run_test'+str(seed)+str(ep)+'.pt'))
    
    def save(self, filepath, ep, seed):
        torch.save(self.pi.state_dict(), f'{filepath}/actor_run_test'+str(seed)+str(ep)+'.pt')
        torch.save(self.q.state_dict(), f'{filepath}/critic_run_test'+str(seed)+str(ep)+'.pt')
