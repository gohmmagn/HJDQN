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

    def mergeLogs(self, modelName, logType, parFileId):
        log_diretory_path = "outputs/{}/{}/{}".format(self.envId, modelName, logType)
        logFiles = []
        for logFile in os.listdir(log_diretory_path):
            logFiles.append(logFile)
        logFiles.remove("fileDirectory.json")
        logArgs = [[logFile, re.split('_|[.]', logFile)] for logFile in logFiles]
        index = [int(logArg[1][-2]) for logArg in logArgs]
        combined = list(zip(index, logArgs))
        combined.sort(key=lambda x: x[0])
        logArgs = [x[1] for x in combined]
        logs = [logArg[0] for logArg in logArgs if logArg[1][2]==str(parFileId)]
        if len(logs) > 1:
            total_log_path = min(logs, key=len)
            try:
                df = pd.read_csv(path.join(log_diretory_path,logs[0]), header=None)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            for i in range(1,len(logs)):
                try:
                    data = pd.read_csv(path.join(log_diretory_path,logs[i]), header=None)
                    df = pd.concat([df, data], axis=0)
                except pd.errors.EmptyDataError:
                    pass
            df.to_csv(path.join(log_diretory_path, total_log_path), index=False, header=None)
            for i in range(1,len(logs)):
                os.remove(path.join(log_diretory_path,logs[i]))

    def checkForParameterFiles(self, modelName):
        parameter_directory_path = "outputs/{}/{}/parameter".format(self.envId, modelName)
        parFileCount = len(os.listdir(parameter_directory_path))
        return parFileCount

    def getModelsOfEnvironment(self):
        environment_folder_path = "outputs/{}".format(self.envId)
        ls = []
        for filefolder in os.listdir(environment_folder_path):
          ls.append(filefolder)
        df_files = pd.DataFrame(ls, columns = ['Model name'])
        return df_files

    def mergeAllLogsOfEnvironment(self):
        df_modelNames = self.getModelsOfEnvironment()
        for i in range(0,df_modelNames.shape[0]):
            for j in range(0,self.checkForParameterFiles(df_modelNames['Model name'][i])):
                self.mergeLogs(df_modelNames['Model name'][i], 'eval_log', j)
                self.mergeLogs(df_modelNames['Model name'][i], 'train_log', j)

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