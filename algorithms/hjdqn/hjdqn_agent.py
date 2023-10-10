import copy
import torch
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss
import gymnasium as gym
from algorithms.utils import freeze
from algorithms.buffer import ReplayBuffer
from algorithms.model import Critic_NN1, Critic_NN2

class HJDQNAgent:
    def __init__(self,
                 dimS,
                 dimA,
                 ctrl_range,
                 gamma,
                 h,
                 L,
                 sigma,
                 verboseLoopTraining,
                 model,
                 acctuator,
                 resortIndex,
                 lr,
                 polyak,
                 buffer_size,
                 batch_size,
                 smooth=False,
                 device='cpu',
                 double=True,
                 scale_factor=1.0,
                 render=False):
        """
        :param dimS: dimension of observation space
        :param dimA: dimension of action space
        :param ctrl_range: range of valid action range
        description of the rest of the params are given in hjdqn.py
        """

        self.dimS = dimS
        self.dimA = dimA
        self.ctrl_range = ctrl_range

        self.h = h

        if model=='Critic_NN1':
          self.Q = Critic_NN1(dimS, dimA, acctuator).to(device)
        if model=='Critic_NN2':
          self.Q = Critic_NN2(dimS, dimA, acctuator, resortIndex).to(device)

        self.target_Q = copy.deepcopy(self.Q).to(device)    # set target network

        if double:
            # In double Q-learning setting, freeze the target network
            freeze(self.target_Q)   # freeze target network in double Q learning setting

        self.optimizer = Adam(self.Q.parameters(), lr=lr)

        self.gamma = gamma
        self.polyak = polyak
        self.L = L

        self.sigma = sigma

        self.buffer = ReplayBuffer(dimS, dimA, buffer_size)
        self.batch_size = batch_size

        self.smooth = smooth

        self.double = double

        self.device = device
        self.render = render
        self.scale_factor = scale_factor
        self.verbose = verboseLoopTraining

        return

    def target_update(self):
        # soft target update
        for p, target_p in zip(self.Q.parameters(), self.target_Q.parameters()):
            target_p.data.copy_(self.polyak * p.data + (1.0 - self.polyak) * target_p.data)

        return

    def get_action(self, state, action, noise):
        """

        :param state: current state
        :param action: current action
        :param explore: bool type variable to indicate exploration
        :return: next action
        """
        device = self.device

        dimS = self.dimS
        dimA = self.dimA

        s = torch.tensor(state, dtype=torch.float).view(1, dimS).to(device)
        a = torch.tensor(action, dtype=torch.float).view(1, dimA).to(device)
        a.requires_grad_(True)

        q = self.Q(s, a)
        q.backward()
        dq = a.grad     # compute gradient of Q w.r.t. a

        with torch.no_grad():
            n = torch.norm(dq)

            # compute increment of action
            if self.smooth:
                a_dot = (self.h * self.L * torch.tanh(n / self.L) / (n + 1e-8)) * dq
                #a_dot = self.h * torch.tanh(1) * dq
            else:
                a_dot = (self.h * self.L / (n + 1e-8)) * dq
                #a_dot = self.h * dq
        a_dot = a_dot.cpu().detach().numpy()
        a_dot = a_dot.reshape(action.shape)

        return np.clip(action + a_dot + noise, -self.ctrl_range, self.ctrl_range)

    def train(self, t, max_iter):

        device = self.device
        gamma = self.gamma
        h = self.h

        L = self.L

        # sample mini-batch
        lim = torch.Tensor(self.ctrl_range).to(device)

        batch = self.buffer.sample_batch(self.batch_size)

        # unroll batch
        with torch.no_grad():
            observations = torch.tensor(batch.state, dtype=torch.float).to(device)
            actions = torch.tensor(batch.act, dtype=torch.float).to(device)
            rewards = torch.tensor(batch.rew, dtype=torch.float).to(device)
            next_observations = torch.tensor(batch.next_state, dtype=torch.float).to(device)
            terminals = torch.tensor(batch.done, dtype=torch.float).to(device)

            mask = 1.0 - terminals

        # Since we need \nabla_a Q(x, a ; \theta), we need to backpropagate the derivatives through action batches.
        actions.requires_grad_(True)

        if self.double:
            # In double Q-learning setting, the control b is chosen using the Q-network instead of the target Q-network
            q = self.Q(observations, actions)
        else:
            q = self.target_Q(observations, actions)

        q.sum().backward()
        # compute sample-wise gradient of Q w.r.t. a
        # here, we use double Q-learning
        g = actions.grad

        with torch.no_grad():
            # divide norm of grad by h in advance, just for computational efficiency
            norm = (torch.sqrt(torch.sum(g**2, dim=1, keepdim=True)) + 1e-9) / (h * L)
            # compute increment of action in sample-wise manner
            da = g / norm
            # next_a = torch.clamp(actions + da, -self.ctrl_range, self.ctrl_range)
            next_a = torch.min(torch.max(actions + da, -lim), lim)
            # target construction
            target = h * rewards + gamma * mask * self.target_Q(next_observations, next_a)

        out = self.Q(observations, actions)

        loss_ftn = MSELoss()
        loss = loss_ftn(out, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_update()

        return

    def eval(self, test_env, t, eval_num=1):
        """
        evaluation of agent
        during evaluation, agent execute noiseless actions
        """

        log = []
        exact_log = []
        for ep in range(eval_num):
            state, _ = test_env.reset()
            exact_state = state

            noise = np.zeros(self.dimA)
            action = test_env.unwrapped.initial_action

            step_count = 0

            ep_reward = 0
            exact_ep_reward = 0

            done = False

            while not done:

                if self.render and ep == 0:
                    test_env.render()

                action = self.get_action(state, action, noise) # deterministic action
                next_state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                step_count += 1
                state = next_state
                ep_reward += reward

                if test_env.unwrapped.useExactSolution:
                   exact_action = test_env.get_exact_action(exact_state) # exact action
                   next_exact_state, exact_reward, _, _, _ = test_env.exact_step(exact_action)
                   exact_state = next_exact_state
                   exact_ep_reward += exact_reward

            if self.render and ep == 0:
                test_env.close()

            log.append(ep_reward)

            if test_env.unwrapped.useExactSolution:
               exact_log.append(exact_ep_reward)

        # normalize score w.r.t. h for consistent return
        avg = self.scale_factor * sum(log) / eval_num

        if test_env.unwrapped.useExactSolution:
           exact_avg = self.scale_factor * sum(exact_log) / eval_num
        else:
           exact_avg = 0.

        if self.verbose:
          print('step {} : avg: {} ; exact_avg: {}'.format(t, avg, exact_avg))

        return [t, avg, exact_avg]

    def save_model(self, path):
        print('adding checkpoint...')
        checkpoint_path = path + '.pth.tar'
        torch.save(
                    {
                     'critic': self.Q.state_dict(),
                     'target_critic': self.target_Q.state_dict(),
                     'critic_optimizer': self.optimizer.state_dict()
                    },
                    checkpoint_path)

        return

    def load_model(self, path, resumeTraining=False):
        print('networks loading...')
        checkpoint = torch.load(path, map_location=self.device)

        self.Q.load_state_dict(checkpoint['critic'])
        self.target_Q.load_state_dict(checkpoint['target_critic'])

        if resumeTraining:
           self.optimizer.load_state_dict(checkpoint['critic_optimizer'])

        return
