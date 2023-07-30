import numpy as np
import os
import json
from os import path
import pandas as pd
import re
import sys
from datetime import datetime

class gdriveFileHandler():

    def __init__(self, envId):
       self.envId = envId
       self.mainDirectory = path.dirname(path.dirname(__file__))

    def mergeEvallogs(self, modelName, parFileId):
        evallog_diretory_path = "outputs/{}/{}/eval_log".format(self.envId, modelName)
        evallogFiles = []
        for evallogFile in os.listdir(evallog_diretory_path):
          evallogFiles.append(evallogFile)
        evallogFiles.remove("fileDirectory.json")
        evallogArgs = [[evallogFile, re.split('_|[.]', evallogFile)] for evallogFile in evallogFiles]
        index = [int(evallogArg[1][-2]) for evallogArg in evallogArgs]
        combined = list(zip(index, evallogArgs))
        combined.sort(key=lambda x: x[0])
        evallogArgs = [x[1] for x in combined]
        evallogs = [evallogArg[0] for evallogArg in evallogArgs if evallogArg[1][2]==str(parFileId)]
        total_evallog_path = min(evallogs, key=len)
        df = pd.read_csv(path.join(evallog_diretory_path,evallogs[0]), header=None)
        for i in range(1,len(evallogs)):
          data = pd.read_csv(path.join(evallog_diretory_path,evallogs[i]), header=None)
          df = pd.concat([df, data], axis=0)
        df.to_csv(path.join(evallog_diretory_path, total_evallog_path), index=False)
        for i in range(1,len(evallogs)):
          os.remove(path.join(evallog_diretory_path,evallogs[i]))

    def getModelsOfEnvironment(self):
        environment_folder_path = "outputs/{}".format(self.envId)
        ls = []
        for filefolder in os.listdir(environment_folder_path):
          ls.append(filefolder)
        df_files = pd.DataFrame(ls, columns = ['Model name'])
        return df_files

    def getRicattiSolutionFiles(self):
        env_data_ricatti_solution_path = "gym_lqr/gym_lqr/envs/data/Ricatti_solution_matrices"
        ricatti_solution_fileDirectory = path.join(env_data_ricatti_solution_path,'ricatti_solution_dictonary.csv')
        df_files = pd.read_csv(ricatti_solution_fileDirectory)
        return df_files

    def getCheckpointFiles(self, modelName):
        checkpoints_diretory_path = "outputs/{}/{}/checkpoints".format(self.envId, modelName)
        checkpoints_fileDirectory = path.join(checkpoints_diretory_path,'fileDirectory.json')
        with open(checkpoints_fileDirectory,'r') as f:
             data = json.loads(f.read())
        df_files = pd.json_normalize(data['fileNames'])
        return df_files

    def getTrainLogFiles(self, modelName):
        trainlog_diretory_path = "outputs/{}/{}/train_log".format(self.envId, modelName)
        trainlog_fileDirectory = path.join(trainlog_diretory_path,'fileDirectory.json')
        with open(trainlog_fileDirectory,'r') as f:
             data = json.loads(f.read())
        df_files = pd.json_normalize(data['fileNames'])
        return df_files

    def getEvalLogFiles(self, modelName):
        evallog_diretory_path = "outputs/{}/{}/eval_log".format(self.envId, modelName)
        evallog_fileDirectory = path.join(evallog_diretory_path,'fileDirectory.json')
        with open(evallog_fileDirectory,'r') as f:
             data = json.loads(f.read())
        df_files = pd.json_normalize(data['fileNames'])
        return df_files

    def getModelParameters(self, modelName, parFileId):
        parameter_directory_path = "outputs/{}/{}/parameter".format(self.envId, modelName)
        with open(path.join(parameter_directory_path,"{}_{}.txt".format(modelName, str(parFileId)))) as f:
             lines = f.readlines()
        parameter_list = [x[:len(x)-1].split(" = ") for x in lines]
        return parameter_list