import csv
import os
from os import path
import json
import pandas as pd
import time
import gymnasium as gym
from algorithms.hjdqn.hjdqn_agent import HJDQNAgent
from algorithms.utils import set_log_dir, get_env_spec, scaled_env
from algorithms.noise import IndependentGaussian, Zero, SDE
import gym_lqr

def run_hjdqn(env_id,
              loadModel='empty',
              resumeTraining=False,
              verboseLoopTraining=True,
              useExactSolution=False,
              useMatCARE=False,
              Kpath='empty',
              model='Critic_NN1',
              L=10.0,
              T=2.0,
              time_steps=200,
              gamma=0.99,
              lr=1e-3,
              sigma=0.15,
              polyak=1e-3,
              max_iter=1e6,
              num_checkpoints=10,
              buffer_size=1e6,
              fill_buffer=20000,
              batch_size=128,
              train_interval=50,
              start_train=10000,
              eval_interval=2000,
              smooth=False,
              double=True,
              noise='none',
              h_scale=1.0,
              device='cpu',
              render=False,
              ):
    """
    param env_id: registered id of the environment
    param L: size of control constraint
    param gamma: discount factor, corresponds to 1 - gamma * h
    param lr: learning rate of optimizer
    param sigma: noise scale of Gaussian noise
    param polyak: target smoothing coefficient
    param hidden1: number of nodes of hidden layer1 of critic
    param hidden2: number of nodes of hidden layer2 of critic
    param max_iter: total number of environment interactions
    param buffer_size: size of replay buffer
    param fill_buffer: number of execution of random policy
    param batch_size: size of minibatch to be sampled during training
    param train_interval: length of interval between consecutive training
    param start_train: the beginning step of training
    param eval_interval: length of interval between evaluation
    param h_scale: scale of timestep of environment
    param device: device used for training
    param render: bool type variable for rendering
    """
    args = locals()

    max_iter = int(max_iter)
    buffer_size = int(buffer_size)
    num_checkpoints = int(num_checkpoints)
    checkpoint_interval = max_iter // (num_checkpoints - 1)
    time_steps = int(time_steps)

    # Create environment
    # Adjust time step length, episode length if needed
    env = scaled_env(env_id=env_id, T=T, time_steps=time_steps, useExactSolution=useExactSolution, useMatCARE=useMatCARE, Kpath=Kpath, scale_factor=h_scale)
    test_env = scaled_env(env_id=env_id, T=T, time_steps=time_steps, useExactSolution=useExactSolution, useMatCARE=useMatCARE, Kpath=Kpath, scale_factor=h_scale)

    acctuator = env.unwrapped.acctuator
    resortIndex = env.unwrapped.resortIndex

    max_ep_len = env.unwrapped._max_episode_steps

    dimS, dimA, h, ctrl_range = get_env_spec(env)

    print('-' * 80)
    print('observation dim : {} / action dim : {}'.format(dimS, dimA))
    print('dt : {}'.format(h))
    print('L : {}'.format(L))
    print('tau : {}'.format(polyak))
    print('lr : {}'.format(lr))
    print('sigma : {}'.format(sigma))
    print('control range : {}'.format(ctrl_range))
    print('-' * 80)

    # Scale gamma & learning rate
    gamma = 1. - h_scale * (1. - gamma)
    lr = h_scale * lr

    # Create agent.
    agent = HJDQNAgent(dimS, dimA, ctrl_range,
                       gamma,
                       h, L, sigma,
                       verboseLoopTraining,
                       model,
                       acctuator,
                       resortIndex,
                       lr,
                       polyak,
                       buffer_size,
                       batch_size,
                       smooth=smooth,
                       device=device,
                       double=double,
                       render=render,
                       scale_factor=h_scale)

    # Setup project directory.
    mainDirectory = path.dirname(path.dirname(path.dirname(__file__)))

    modelName = None
    new_parfile_id = None
    resumeTrainingFlag = None
    currentIteration = 0
    fileNumber = 0

    if loadModel=='empty':
       current_time = time.strftime("%Y-%m-%dT%H%M%S")
       modelName = 'HJDQN_' + current_time

    if loadModel!='empty':
       modelName = 'HJDQN_' + loadModel.split('_')[1]
       resumeTrainingFlag = resumeTraining
       agent.load_model(path.join(mainDirectory, "outputs/{}/{}/checkpoints/{}".format(env_id,modelName,loadModel)),resumeTraining=resumeTraining)

    set_log_dir(env_id, modelName)

    parameter_directory_path = path.join(mainDirectory, "outputs/{}/{}/parameter".format(env_id,modelName))
    parameter_directory_files = os.listdir(parameter_directory_path)

    if not parameter_directory_files:
       new_parfile_id = 0
       with open(path.join(parameter_directory_path,'{}_{}.txt'.format(modelName, str(new_parfile_id))), 'w') as f:
            print('dimS','=',dimS,file=f)
            print('dimA','=',dimA,file=f)
            print('h','=',h,file=f)
            print('ctrl_range','=',ctrl_range,file=f)
            for key, val in args.items():
                print(key, '=', val, file=f)
    else:
       tmp_fileName = '{}_tmp.txt'.format(modelName)

       with open(path.join(parameter_directory_path,tmp_fileName), 'w') as f:
            print('dimS','=',dimS,file=f)
            print('dimA','=',dimA,file=f)
            print('h','=',h,file=f)
            print('ctrl_range','=',ctrl_range,file=f)
            for key, val in args.items():
                print(key, '=', val, file=f)

       parameter_directory_files_identity = [x.split('_')[2].split('.txt')[0] for x in parameter_directory_files if x.split('_')[2].split('.txt')[0] != 'tmp']

       eq_count = 0

       for ids in parameter_directory_files_identity:
           with open(path.join(parameter_directory_path,tmp_fileName)) as f1:
                lines_tmp = f1.readlines()
           with open(path.join(parameter_directory_path,'{}_{}.txt'.format(modelName,ids))) as f2:
                lines = f2.readlines()
           if len(lines_tmp) == len(lines):
              for line1 in lines_tmp:
                  for line2 in lines:
                      if line1 == line2:
                         eq_count += 1
           if eq_count == len(lines):
              new_parfile_id = ids
              break
           else:
              eq_count = 0

       if new_parfile_id==None:
          max_parfile_id = max([int(ids) for ids in parameter_directory_files_identity])
          new_parfile_id = max_parfile_id + 1
          os.rename(path.join(parameter_directory_path,tmp_fileName),path.join(parameter_directory_path,'{}_{}.txt'.format(modelName,str(new_parfile_id))))
       else:
          os.remove(path.join(parameter_directory_path,tmp_fileName))

    train_log_directory_path = path.join(mainDirectory, "outputs/{}/{}/train_log".format(env_id,modelName))
    train_log_files = os.listdir(train_log_directory_path)
    train_log_path = None

    eval_log_directory_path = path.join(mainDirectory, "outputs/{}/{}/eval_log".format(env_id,modelName))
    eval_log_files = os.listdir(eval_log_directory_path)
    eval_log_path = None

    checkpoint_directory_path = path.join(mainDirectory, "outputs/{}/{}/checkpoints".format(env_id,modelName))
    checkpoint_files = os.listdir(checkpoint_directory_path)
    checkpoint_path = None

    trainlog_fileDirectory_path = path.join(mainDirectory, "outputs/{}/{}/train_log/fileDirectory.json".format(env_id,modelName))
    evallog_fileDirectory_path = path.join(mainDirectory, "outputs/{}/{}/eval_log/fileDirectory.json".format(env_id,modelName))
    checkpoints_fileDirectory_path = path.join(mainDirectory, "outputs/{}/{}/checkpoints/fileDirectory.json".format(env_id,modelName))

    def write_init_json(init_json, json_filename):
        with open(json_filename,'w') as file:
             json.dump(init_json, file)

    def write_json(fileNameData, json_filename):
        with open(json_filename,'r+') as file:
             file_data = json.load(file)
             file_data["fileNames"].append(fileNameData)
             file.seek(0)
             json.dump(file_data, file)

    if not train_log_files:
       init_json = {"modelName": modelName, "fileNames": []}

       write_init_json(init_json, trainlog_fileDirectory_path)
       write_init_json(init_json, evallog_fileDirectory_path)
       write_init_json(init_json, checkpoints_fileDirectory_path)

       train_log_path = '{}_{}.csv'.format(modelName, str(fileNumber))
       eval_log_path = '{}_{}.csv'.format(modelName, str(fileNumber))

       fileNameData_train = { "fileName": train_log_path, "parFileId": new_parfile_id, "resumeTrainingFlag": resumeTrainingFlag, "Iteration": currentIteration}
       fileNameData_eval = { "fileName": eval_log_path, "parFileId": new_parfile_id, "resumeTrainingFlag": resumeTrainingFlag, "Iteration": currentIteration}

       write_json(fileNameData_train, trainlog_fileDirectory_path)
       write_json(fileNameData_eval, evallog_fileDirectory_path)
    else:
       pureTrainLogs = [tl for tl in train_log_files if tl!='fileDirectory.json']
       fileNumber = max([int(x.split('_')[2].split('.csv')[0]) for x in pureTrainLogs]) + 1
       with open(checkpoints_fileDirectory_path,'r') as f:
            data = json.loads(f.read())
       df_files = pd.json_normalize(data['fileNames'])
       modelFileInfo = df_files[df_files['fileName'] == loadModel]
       modelIteration = modelFileInfo['Iteration'].tolist()[0]
       currentIteration = modelIteration
       train_log_path = '{}_{}.csv'.format(modelName, str(fileNumber))
       eval_log_path = '{}_{}.csv'.format(modelName, str(fileNumber))
       fileNameData_train = { "fileName": train_log_path, "parFileId": new_parfile_id, "resumeTrainingFlag": resumeTrainingFlag, "Iteration": currentIteration}
       fileNameData_eval = { "fileName": eval_log_path, "parFileId": new_parfile_id, "resumeTrainingFlag": resumeTrainingFlag, "Iteration": currentIteration}
       write_json(fileNameData_train, trainlog_fileDirectory_path)
       write_json(fileNameData_eval, evallog_fileDirectory_path)

    def createLogger(train_log_path, eval_log_path):
      train_log = open(path.join(train_log_directory_path, train_log_path),
                       'w',
                       encoding='utf-8',
                       newline='')

      eval_log = open(path.join(eval_log_directory_path, eval_log_path),
                      'w',
                      encoding='utf-8',
                      newline='')	

      train_logger = csv.writer(train_log)
      eval_logger = csv.writer(eval_log)
      return train_logger, eval_logger, train_log, eval_log

    def closeLogs(train_log, eval_log):
      train_log.close()
      eval_log.close()

    train_logger, eval_logger, train_log, eval_log = createLogger(train_log_path, eval_log_path)

    # set noise process for exploration
    if noise == 'gaussian':
        noise_process = IndependentGaussian(dim=dimA, sigma=sigma)
    elif noise == 'sde':
        print('noise set to Ornstein-Uhlenbeck process')
        noise_process = SDE(dim=dimA, sigma=sigma, dt=h)
    elif noise == 'none':
        print('noise process is set to zero')
        noise_process = Zero(dim=dimA)
    else:
        print('unidentified noise type : noise process is set to zero')
        noise_process = Zero(dim=dimA)

    # start environment roll-out
    state, _ = env.reset()
    noise = noise_process.reset()
    step_count = 0
    ep_reward = 0.

    action = env.unwrapped.action_space.sample()

    # Log file number
    k = 1

    # main loop
    for t in range(max_iter + 1):

        # t : number of env-agent interactions (=number of transition samples observed)
        if t < fill_buffer:
            # first collect sufficient number of samples during the initial stage
            action = env.unwrapped.action_space.sample()
        else:
            action = agent.get_action(state, action, noise)
            noise = noise_process.sample()

        next_state, reward, terminated, truncated, _ = env.step(action)  # env-agent interaction
        done = terminated or truncated
        step_count += 1

        if step_count == max_ep_len:
            # when the episode roll-out is truncated artificially(so that done=True), set done=False
            # thus, done=True only if the state is a terminal state
            done = False

        agent.buffer.append(state, action, reward, next_state, done)    # save the transition sample

        ep_reward += reward
        state = next_state

        if done or (step_count == max_ep_len):
            train_logger.writerow([t, ep_reward])

            # restart an episode
            state, _ = env.reset()
            noise = noise_process.reset()
            action = env.unwrapped.action_space.sample()
            step_count = 0
            ep_reward = 0.

        # Start training after sufficient number of transition samples are gathered
        if (t >= start_train) and (t % train_interval == 0):
            for _ in range(train_interval):
                agent.train(t, max_iter)

        if t % eval_interval == 0:
            eval_data = agent.eval(test_env, t)
            eval_logger.writerow(eval_data)

        if t % checkpoint_interval == 0:
           closeLogs(train_log, eval_log)
           train_log_path = '{}_{}_{}.csv'.format(modelName, str(fileNumber), str(k))
           eval_log_path = '{}_{}_{}.csv'.format(modelName, str(fileNumber), str(k))
           train_logger, eval_logger, train_log, eval_log = createLogger(train_log_path, eval_log_path)
           k = k+1
           checkpoint_path = '{}_{}_{}'.format(modelName, str(fileNumber),str(currentIteration+t))
           agent.save_model(path.join(checkpoint_directory_path,checkpoint_path))
           fileNameData_checkpoint = { "fileName": checkpoint_path+".pth.tar", "avg_reward": eval_data[1], "parFileId": new_parfile_id, "resumeTrainingFlag": resumeTrainingFlag, "Iteration": currentIteration+t}
           write_json(fileNameData_checkpoint, checkpoints_fileDirectory_path)         

    print("L: {}, tau: {}, lr: {}, sigma: {}, avg_reward: {}, exact_avg_reward: {}".format(L,polyak,lr,sigma,eval_data[1],eval_data[2]))

    closeLogs(train_log, eval_log)

    return
