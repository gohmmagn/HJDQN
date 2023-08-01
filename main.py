from algorithms.hjdqn.hjdqn import run_hjdqn
from algorithms.ddpg.ddpg import run_ddpg

import argparse
import torch

parser = argparse.ArgumentParser()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser.add_argument('--env', required=True)
parser.add_argument('--algo', required=False, choices=['hjdqn', 'ddpg'], default='ddpg')
parser.add_argument('--loopTraining', required=False, action='store_true')
parser.add_argument('--loadModel', required=False, default='empty')
parser.add_argument('--resumeTraining', required=False, action='store_true')
parser.add_argument('--useExactSolution', required=False, action='store_true')
parser.add_argument('--useMatCARE', required=False, action='store_true')
parser.add_argument('--Kpath', required=False, default='empty')
parser.add_argument('--model', required=False, default='Critic_NN1')
parser.add_argument('--device', required=False, default=device)
parser.add_argument('--num_trial', required=False, default=1, type=int)
parser.add_argument('--max_iter', required=False, default=1e6, type=float)
parser.add_argument('--num_checkpoints', required=False, default=10, type=int)
parser.add_argument('--eval_interval', required=False, default=2000, type=int)
parser.add_argument('--render', action='store_true')
parser.add_argument('--q_lr', required=False, default=1e-3, type=float)
parser.add_argument('--pi_lr', required=False, default=1e-4, type=float)
parser.add_argument('--L', required=False, default=30.0, type=float)
parser.add_argument('--T', required=False, default=2.0, type=float)
parser.add_argument('--time_steps', required=False, default=200, type=int)
parser.add_argument('--tau', required=False, default=1e-3, type=float)
parser.add_argument('--lr', required=False, default=1e-4, type=float)
parser.add_argument('--noise', required=False, default='none')
parser.add_argument('--sigma', required=False, default=0.1, type=float)
parser.add_argument('--hidden1', required=False, default=256, type=int)
parser.add_argument('--hidden2', required=False, default=256, type=int)
parser.add_argument('--train_interval', required=False, default=50, type=int)
parser.add_argument('--start_train', required=False, default=10000, type=int)
parser.add_argument('--fill_buffer', required=False, default=20000, type=int)
parser.add_argument('--h_scale', required=False, default=1.0, type=float)
parser.add_argument('--batch_size', required=False, default=128, type=int)
parser.add_argument('--gamma', required=False, default=0.99, type=float)
parser.add_argument('--smooth', action='store_true')
parser.add_argument('--no_double', action='store_false')
parser.add_argument('--verboseLoopTraining', required=False, action='store_false')

args = parser.parse_args()

# HJDQN hyperparameter.

L = [5, 10, 15, 20]
tau = [1e-3]
lr = [1e-2, 1e-4]
sigma = [0.1, 0.25, 1]

hyper_pars = [[20, 1e-3, 1e-4, 0.25]]

#for Li in L:
#  for taui in tau:
#    for lri in lr:
#      for sigmai in sigma:
#          hyper_pars.append([Li, taui, lri, sigmai])

if args.algo=='hjdqn':
   if args.loopTraining:
      for hyp in hyper_pars:
          for _ in range(args.num_trial):
              run_hjdqn(args.env,
                      loadModel=args.loadModel,
                      resumeTraining=args.resumeTraining,
                      useExactSolution=args.useExactSolution,
                      useMatCARE=args.useMatCARE,
                      Kpath=args.Kpath,
                      model=args.model,
                      L=hyp[0],
                      T=args.T,
                      time_steps=args.time_steps,
                      gamma=args.gamma,
                      lr=hyp[2],
                      sigma=hyp[3],
                      verboseLoopTraining=args.verboseLoopTraining,
                      polyak=hyp[1],
                      max_iter=args.max_iter,
                      num_checkpoints=args.num_checkpoints,
                      buffer_size=1e5,
                      fill_buffer=args.fill_buffer,
                      train_interval=args.train_interval,
                      start_train=args.start_train,
                      eval_interval=args.eval_interval,
                      h_scale=args.h_scale,
                      device=args.device,
                      double=args.no_double,
                      smooth=args.smooth,
                      noise=args.noise,
                      batch_size=args.batch_size,
                      render=args.render
                      )
   else:
      for _ in range(args.num_trial):
          run_hjdqn(args.env,
                  loadModel=args.loadModel,
                  resumeTraining=args.resumeTraining,
                  useExactSolution=args.useExactSolution,
                  useMatCARE=args.useMatCARE,
                  Kpath=args.Kpath,
                  model=args.model,
                  L=args.L,
                  T=args.T,
                  time_steps=args.time_steps,
                  gamma=args.gamma,
                  lr=args.lr,
                  sigma=args.sigma,
                  verboseLoopTraining=args.verboseLoopTraining,
                  polyak=args.tau,
                  max_iter=args.max_iter,
                  num_checkpoints=args.num_checkpoints,
                  buffer_size=1e5,
                  fill_buffer=args.fill_buffer,
                  train_interval=args.train_interval,
                  start_train=args.start_train,
                  eval_interval=args.eval_interval,
                  h_scale=args.h_scale,
                  device=args.device,
                  double=args.no_double,
                  smooth=args.smooth,
                  noise=args.noise,
                  batch_size=args.batch_size,
                  render=args.render
                  )
else:
   run_ddpg(args.env,
          loadModel=args.loadModel,
          resumeTraining=args.resumeTraining,
          useExactSolution=args.useExactSolution,
          useMatCARE=args.useMatCARE,
          Kpath=args.Kpath,
          model=args.model,
          T=args.T,
          time_steps=args.time_steps,
          gamma=args.gamma,
          actor_lr=args.pi_lr,
          critic_lr=args.q_lr,
          polyak=args.tau,
          sigma=args.sigma,
          hidden_size1=args.hidden1,
          hidden_size2=args.hidden2,
          max_iter=args.max_iter,
          num_checkpoints=args.num_checkpoints,
          eval_interval=args.eval_interval,
          start_train=args.start_train,
          train_interval=args.train_interval,
          buffer_size=1e6,
          fill_buffer=args.fill_buffer,
          batch_size=args.batch_size,
          h_scale=args.h_scale,
          device=args.device,
          render=args.render
          )