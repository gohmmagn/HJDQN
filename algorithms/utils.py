import os
import gymnasium as gym

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def get_env_spec(env):
    dimS = env.unwrapped.observation_space.shape[0]
    dimA = env.unwrapped.action_space.shape[0]
    ctrl_range = env.unwrapped.action_space.high

    return dimS, dimA, env.unwrapped.dt, ctrl_range

def set_log_dir(env_id, modelName):

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')

    if not os.path.exists('./outputs/' + env_id):
        os.mkdir('./outputs/' + env_id)

    if not os.path.exists('./outputs/' + env_id + '/' + modelName):
        os.mkdir('./outputs/' + env_id + '/' + modelName)

    if not os.path.exists('./outputs/' + env_id + '/' + modelName + '/train_log/'):
        os.mkdir('./outputs/' + env_id + '/' + modelName + '/train_log/')
    if not os.path.exists('./outputs/' + env_id + '/' + modelName + '/eval_log/'):
        os.mkdir('./outputs/' + env_id + '/' + modelName + '/eval_log/')
    if not os.path.exists('./outputs/' + env_id + '/' + modelName + '/checkpoints/'):
        os.mkdir('./outputs/' + env_id + '/' + modelName + '/checkpoints/')
    if not os.path.exists('./outputs/' + env_id + '/' + modelName + '/pde_solutions/'):
        os.mkdir('./outputs/' + env_id + '/' + modelName + '/pde_solutions/')
    if not os.path.exists('./outputs/' + env_id + '/' + modelName + '/parameter/'):
        os.mkdir('./outputs/' + env_id + '/' + modelName + '/parameter/')

    return

def scaled_env(env_id, T, time_steps, useExactSolution, useMatCARE, Kpath, scale_factor):
    """
    adjust environment parameters related to time discretization
    """
    env = gym.make(env_id, render_mode='human', T=T, time_steps=time_steps, useExactSolution=useExactSolution, useMatCARE=useMatCARE, Kpath=Kpath)
    env.unwrapped._max_episode_steps = time_steps

    #env.dt *= scale_factor
    #env._max_episode_steps = int(env._max_episode_steps / scale_factor)

    return env
