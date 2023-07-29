import time
import csv
import os
from os import path
import json
import pandas as pd
import gymnasium as gym
import gym_lqr
from algorithms.ddpg.ddpg_agent import DDPGAgent
from algorithms.utils import get_env_spec, set_log_dir

def run_ddpg(env_id,
             loadModel='empty',
             resumeTraining=False,
             useExactSolution=False,
             useMatCARE=False,
             Kpath='empty',
             verboseLoopTraining=True,
             model='Critic_NN1',
             T=2.0,
             time_steps=200,
             gamma=0.99,
             actor_lr=1e-4,
             critic_lr=1e-3,
             polyak=1e-3,
             sigma=0.1,
             hidden_size1=256,
             hidden_size2=256,
             max_iter=1e6,
             num_checkpoints=10,
             eval_interval=2000,
             start_train=10000,
             train_interval=50,
             buffer_size=1e6,
             fill_buffer=20000,
             batch_size=128,
             h_scale=1.0,
             device='cpu',
             render='False'
             ):
    """
    :param env_id: registered id of the environment
    :param gamma: discount factor
    :param actor_lr: learning rate of actor optimizer
    :param critic_lr: learning rate of critic optimizer
    :param sigma: noise scale of Gaussian noise
    :param polyak: target smoothing coefficient
    :param hidden1: number of nodes of hidden layer1 of critic
    :param hidden2: number of nodes of hidden layer2 of critic
    :param max_iter: total number of environment interactions
    :param buffer_size: size of replay buffer
    :param fill_buffer: number of execution of random policy
    :param batch_size: size of minibatch to be sampled during training
    :param train_interval: length of interval between consecutive training
    :param start_train: the beginning step of training
    :param eval_interval: length of interval between evaluation
    :param h_scale: scale of timestep of environment
    :param device: device used for training
    :param render: bool type variable for rendering
    """
    args = locals()

    max_iter = int(max_iter)
    buffer_size = int(buffer_size)
    num_checkpoints = int(num_checkpoints)
    checkpoint_interval = max_iter // (num_checkpoints - 1)
    time_steps = int(time_steps)

    env = gym.make(env_id, render_mode='human', T=T, time_steps=time_steps, useExactSolution=useExactSolution, useMatCARE=useMatCARE, Kpath=Kpath)
    test_env = gym.make(env_id, render_mode='human', T=T, time_steps=time_steps, useExactSolution=useExactSolution, useMatCARE=useMatCARE, Kpath=Kpath)

    acctuator = env.unwrapped.acctuator
    resortIndex = env.unwrapped.resortIndex

    max_ep_len = env.unwrapped._max_episode_steps

    dimS, dimA, h, ctrl_range = get_env_spec(env)

    agent = DDPGAgent(dimS,
                      dimA,
                      ctrl_range,
                      gamma=gamma,
                      actor_lr=actor_lr,
                      critic_lr=critic_lr,
                      polyak=polyak,
                      sigma=sigma,
                      verboseLoopTraining=verboseLoopTraining,
                      model=model,
                      hidden1=hidden_size1,
                      hidden2=hidden_size2,
                      acctuator=acctuator,
                      resortIndex=resortIndex,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      h_scale=h_scale,
                      device=device,
                      render=render)

    # Setup project directory.
    mainDirectory = path.dirname(path.dirname(path.dirname(__file__)))

    modelName = None
    new_parfile_id = None
    resumeTrainingFlag = None
    currentIteration = 0
    fileNumber = 0

    if loadModel=='empty':
       current_time = time.strftime("%Y-%m-%dT%H%M%S")
       modelName = 'DDPG_' + current_time

    if loadModel!='empty':
       modelName = 'DDPG_' + loadModel.split('_')[1]
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

    state, _ = env.reset()
    step_count = 0
    ep_reward = 0

    # main loop
    for t in range(max_iter + 1):
        if t < fill_buffer:
            # first collect sufficient number of samples during the initial stage
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)  # env-agent interaction
        done = terminated or truncated
        step_count += 1

        if step_count == max_ep_len:
            done = False

        agent.buffer.append(state, action, reward, next_state, done)    # save the transition sample

        state = next_state
        ep_reward += reward

        if done or (step_count == max_ep_len):
            train_logger.writerow([t, ep_reward])
            state, _ = env.reset()
            step_count = 0
            ep_reward = 0

        if (t >= start_train) and (t % train_interval == 0):
            # Start training after sufficient number of transition samples are gathered
            for _ in range(train_interval):
                agent.train()

        if t % eval_interval == 0:
            eval_data = agent.eval(test_env, t)
            eval_logger.writerow(eval_data)

        if t % checkpoint_interval == 0:
            checkpoint_path = '{}_{}_{}'.format(modelName, str(fileNumber),str(currentIteration+t))
            agent.save_model(path.join(checkpoint_directory_path,checkpoint_path))
            fileNameData_checkpoint = { "fileName": checkpoint_path+".pth.tar", "avg_reward": eval_data[1], "parFileId": new_parfile_id, "resumeTrainingFlag": resumeTrainingFlag, "Iteration": currentIteration+t}
            write_json(fileNameData_checkpoint, checkpoints_fileDirectory_path)

    print("Actor lr: {}, Critic lr: {}, tau: {}, sigma: {}, avg_reward: {}, exact_avg_reward: {}".format(actor_lr, critic_lr, polyak,sigma, eval_data[1], eval_data[2]))

    train_log.close()
    eval_log.close()

    return

