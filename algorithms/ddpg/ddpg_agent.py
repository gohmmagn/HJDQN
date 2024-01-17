import torch.optim
from torch.nn import MSELoss
import numpy as np
import gym
import copy
from algorithms.buffer import ReplayBuffer
from algorithms.model import Actor, Critic_NN1, Critic_NN2
from algorithms.utils import freeze, unfreeze

class DDPGAgent:
    def __init__(self,
                 dimS,
                 dimA,
                 ctrl_range,
                 gamma=0.99,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 polyak=1e-3,
                 sigma=0.1,
                 verboseLoopTraining=True,
                 model='Critic_NN1',
                 hidden1=400,
                 hidden2=300,
                 acctuator=[],
                 resortIndex=[],
                 buffer_size=int(1e6),
                 batch_size=128,
                 h_scale=1.0,
                 device='cpu',
                 render=False):
        """
        :param dimS: dimension of observation space
        :param dimA: dimension of action space
        :param ctrl_range: range of valid action range
        description of the rest of the params are given in ddpg.py
        """
        self.dimS = dimS
        self.dimA = dimA
        self.ctrl_range = ctrl_range

        self.gamma = gamma
        self.pi_lr = actor_lr
        self.q_lr = critic_lr
        self.polyak = polyak
        self.sigma = sigma

        self.batch_size = batch_size
        # networks definition
        # pi : actor network, Q : critic network
        self.pi = Actor(dimS, dimA, hidden1, hidden2, ctrl_range).to(device)
        if model=='Critic_NN1':
          self.Q = Critic_NN1(dimS, dimA, acctuator, device).to(device)
        if model=='Critic_NN2':
          self.Q = Critic_NN2(dimS, dimA, acctuator, resortIndex, device).to(device)

        # target networks
        self.target_pi = copy.deepcopy(self.pi).to(device)
        self.target_Q = copy.deepcopy(self.Q).to(device)

        freeze(self.target_pi)
        freeze(self.target_Q)

        self.buffer = ReplayBuffer(dimS, dimA, limit=buffer_size)

        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.q_lr)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.pi_lr)

        self.h_scale = h_scale
        self.device = device
        self.render = render
        self.verbose = verboseLoopTraining

    def target_update(self):
        # soft-update for both actors and critics
        # \theta^\prime = \tau * \theta + (1 - \tau) * \theta^\prime
        for p, target_p in zip(self.pi.parameters(), self.target_pi.parameters()):
            target_p.data.copy_(self.polyak * p.data + (1.0 - self.polyak) * target_p.data)

        for params, target_params in zip(self.Q.parameters(), self.target_Q.parameters()):
            target_params.data.copy_(self.polyak * params.data + (1.0 - self.polyak) * target_params.data)

    def get_action(self, state_tensor, eval=False):

        state_arr = state_tensor[np.newaxis]    
        state_tensor = torch.tensor(state_arr, dtype=torch.float).to(self.device)

        with torch.no_grad():
            action = self.pi(state_tensor)
            action = action.cpu().detach().numpy()

            action = np.squeeze(action, axis=0)

        if not eval:
            noise = self.sigma * np.random.randn(self.dimA)
            return np.clip(action + noise, -self.ctrl_range, self.ctrl_range)
        else:
            return action

    def train(self):
        """
        train actor-critic network using DDPG
        """
        device = self.device

        batch = self.buffer.sample_batch(batch_size=self.batch_size)

        # unroll batch
        observations = torch.tensor(batch.state, dtype=torch.float).to(device)
        actions = torch.tensor(batch.act, dtype=torch.float).to(device)
        rewards = torch.tensor(batch.rew, dtype=torch.float).to(device)
        next_observations = torch.tensor(batch.next_state, dtype=torch.float).to(device)
        terminals = torch.tensor(batch.done, dtype=torch.float).to(device)

        mask = 1 - terminals

        # compute TD targets based on target networks
        # if done, set target value to reward
        with torch.no_grad():
            target = rewards + self.gamma * mask * self.target_Q(next_observations, self.target_pi(next_observations))

        out = self.Q(observations, actions)
        loss_ftn = MSELoss()
        loss = loss_ftn(out, target)
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()

        freeze(self.Q)
        pi_loss = - torch.mean(self.Q(observations, self.pi(observations)))
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()
        unfreeze(self.Q)

        self.target_update()

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

            step_count = 0

            ep_reward = 0
            exact_ep_reward = 0

            done = False

            while not done:
                if self.render and ep == 0:
                    test_env.render()

                action = self.get_action(state, eval=True)
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
        avg = sum(log) / eval_num

        if test_env.unwrapped.useExactSolution:
           exact_avg = sum(exact_log) / eval_num
        else:
           exact_avg = 0.

        if self.verbose:
          print('step {} : avg: {} ; exact_avg: {}'.format(t, avg, exact_avg))

        return [t, avg, exact_avg]


    def save_model(self, path):
        print('adding checkpoints...')
        checkpoint_path = path + '.pth.tar'
        torch.save(
                    {'actor': self.pi.state_dict(),
                     'critic': self.Q.state_dict(),
                     'target_actor': self.target_pi.state_dict(),
                     'target_critic': self.target_Q.state_dict(),
                     'actor_optimizer': self.pi_optimizer.state_dict(),
                     'critic_optimizer': self.Q_optimizer.state_dict()
                    },
                    checkpoint_path)

        return

    def load_model(self, path, resumeTraining=False):
        print('networks loading...')
        checkpoint = torch.load(path, map_location=self.device)

        self.pi.load_state_dict(checkpoint['actor'])
        self.Q.load_state_dict(checkpoint['critic'])
        self.target_pi.load_state_dict(checkpoint['target_actor'])
        self.target_Q.load_state_dict(checkpoint['target_critic'])

        if resumeTraining:
           self.pi_optimizer.load_state_dict(checkpoint['actor_optimizer'])
           self.Q_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        return
