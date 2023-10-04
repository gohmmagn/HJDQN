import tkinter
import os
from tkinter import Listbox, Button, Frame, ttk, messagebox
from os import path
import matplotlib.pyplot as plt
from  matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np

def readSubDirNamesAndPaths(folderDir):
    subDirNames = []
    subDirPaths = []
    for file in os.listdir(folderDir):
        d = os.path.join(folderDir, file)
        if os.path.isdir(d):
            _, basename = os.path.split(d)
            subDirNames.append(basename)
            subDirPaths.append(d)
    return subDirNames, subDirPaths

def getParFiles(parDirPaths):
    parFileNames = []
    parFilePaths = []
    for file in os.listdir(parDirPaths):
        d = os.path.join(parDirPaths, file)
        _, basename = os.path.split(d)
        parFileNames.append(basename)
        parFilePaths.append(d)
    return parFileNames, parFilePaths

def evallogPlotDiffApproxExact(ax, evallog_path):
    avgCostEvalAndExact = pd.read_csv(evallog_path).values
    return ax.plot(avgCostEvalAndExact[:,0], np.abs(avgCostEvalAndExact[:,1]-avgCostEvalAndExact[:,2]))

def evallogPlotLogQuotApproxExact(ax, evallog_path):
    avgCostEvalAndExact = pd.read_csv(evallog_path).values
    return ax.plot(avgCostEvalAndExact[:,0], np.log(avgCostEvalAndExact[:,1]/avgCostEvalAndExact[:,2]))

def evallogPlotAbs(ax, evallog_path):
    avgCostEval = pd.read_csv(evallog_path).values
    return ax.plot(avgCostEval[:,0], np.abs(avgCostEval[:,1]))

def evallogPlotLog(ax, evallog_path):
    avgCostEval = pd.read_csv(evallog_path).values
    return ax.plot(avgCostEval[:,0], np.log(np.abs(avgCostEval[:,1])))

def trainlogPlotAbs(ax, trainlog_path):
    epRewardTraining = pd.read_csv(trainlog_path).values
    return ax.plot(epRewardTraining[:,0], np.abs(epRewardTraining[:,1]))

def trainlogPlotLog(ax, trainlog_path):
    epRewardTraining = pd.read_csv(trainlog_path).values
    return ax.plot(epRewardTraining[:,0], np.log(np.abs(epRewardTraining[:,1])))

def createWidgets(envName, title, xlabel, ylabel, filepaths, type):
        
    widget = tkinter.Tk()
    widget.title(title)
    widget.geometry("650x650")

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            widget.destroy()

    widget.protocol("WM_DELETE_WINDOW", on_closing)

    f0 = tkinter.Frame(widget)
    fig = plt.figure(figsize=(8, 8))
   
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    for i in range(0,len(filepaths)):
        if envName == 'Linear1dPDEEnv-v0' or envName == 'Linear2dPDEEnv-v0':
            if type == 'evalExactAndApproxDiff':
                evallogPlotDiffApproxExact(ax1, filepaths[i])
            if type == 'evalExactAndApproxLogQuot':
                evallogPlotLogQuotApproxExact(ax1, filepaths[i])
        if envName == 'NonLinearPDEEnv-v0':
            if type == 'evalAbs':
                evallogPlotAbs(ax1, filepaths[i])
            if type == 'evalLogAbs':
                evallogPlotLog(ax1, filepaths[i])
        if type == 'trainAbs':
            trainlogPlotAbs(ax1, filepaths[i])
        if type == 'trainLogAbs':
            trainlogPlotLog(ax1, filepaths[i])

    canvas = FigureCanvasTkAgg(fig, f0)
    toolbar = NavigationToolbar2Tk(canvas, f0)
    toolbar.update()
    canvas._tkcanvas.pack(fill=tkinter.BOTH, expand=1)
    f0.pack(fill=tkinter.BOTH, expand=1)

# App frames
app = tkinter.Tk()
app.title("ResultViewer")
app.geometry('800x450')
app.columnconfigure(0, weight=1)
app.rowconfigure(0, weight=1)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)

Lb_frame = Frame(app)
Lb_frame.grid(row=0, column=0, padx=10, pady=5)

LbEnvs_frame = Frame(Lb_frame)
LbEnvs_frame.grid(row=0, column=0, padx=10, pady=5)

LbModels_frame = Frame(Lb_frame)
LbModels_frame.grid(row=0, column=1, padx=10, pady=5)

Parameter_frame = tkinter.LabelFrame(app ,text='Parameter')
Parameter_frame.columnconfigure(0, weight=1)
Parameter_frame.rowconfigure(0, weight=1)
Parameter_frame.grid(column=0,row=1,sticky='nsew',padx=6,pady=6)

DeleteBtn_Frame = Frame(Parameter_frame)
DeleteBtn_Frame.grid(row=2, column=0, padx=10, pady=5)

# UI elements

# Outputs folder
mainDir = path.dirname(path.dirname(__file__))
outputsDir = path.join(mainDir, "outputs")

envNames, envFolderPaths = readSubDirNamesAndPaths(outputsDir)
LbEnvs = Listbox(LbEnvs_frame, selectmode=tkinter.SINGLE)
for i in range(0,len(envNames)):
    LbEnvs.insert(i, envNames[i])
LbEnvs.grid(row=0, column=0, padx=10, pady=5)

LbModels = Listbox(LbModels_frame, selectmode=tkinter.MULTIPLE, width = 25)
LbModels.grid(row=0, column=1, padx=10, pady=5)

envDirName = []
parFilePaths = []
parNames = ['ParFile', 'dimS', 'dimA', 'h', 'ctrl_range', 'env_id', 'loadModel', 'resumeTraining', 'verboseLoopTraining', 'useExactSolution', 'useMatCARE', 'Kpath', 'model', 'L', 'T', 'time_steps', 'gamma', 'lr', 'sigma', 'polyak', 'max_iter', 'num_checkpoints', 'buffer_size', 'fill_buffer', 'batch_size', 'train_interval', 'start_train', 'eval_interval', 'smooth', 'double', 'noise', 'h_scale', 'device', 'render']
tree = ttk.Treeview(Parameter_frame, columns=parNames, show='headings', height=5, selectmode='extended')

def selected_env():
    global LbEnvs, LbModels, envDirName, envFolderPaths, tree
    for item in tree.get_children():
            tree.delete(item)
    cursor =  LbEnvs.curselection()
    envDirName = []
    if len(cursor)!=0:
        envDir = envFolderPaths[cursor[0]]
        envDirName = envDir
        modelNames, _ = readSubDirNamesAndPaths(envDir)
        dates = [models.split("_")[1].split("T")[0] for models in modelNames]
        dates = [date[8:10]+"."+date[5:7]+"."+date[2:4] for date in dates]
        times = [models.split("_")[1].split("T")[1] for models in modelNames]
        times = [time[0:2]+":"+time[2:4]+":"+time[4:6] for time in times]
        listitem = ["HJDQN "+info[0]+" "+info[1] for info in zip(dates, times) ]
        LbModels.delete(0,tkinter.END)
        for i in range(0,len(listitem)):
            LbModels.insert(i, listitem[i])
 
envBtn = Button(LbEnvs_frame, text='Print Selected', command=selected_env)
envBtn.grid(row=1, column=0, padx=10, pady=5)

def selected_models():
    global LbModels, envDirName, parFilePaths, tree
    cursor =  LbModels.curselection()
    tableContentZero = [tree.set(item,0) for item in tree.get_children()]
    parFilePaths = []
    parNames = []
    parArgs = []
    if len(cursor)!=0:
        _, parDirPaths = readSubDirNamesAndPaths(envDirName)
        parDirPaths = [path.join(modelPath,"parameter") for modelPath in parDirPaths]
        selectedParDirPaths = [parDirPaths[i] for i in cursor]
        for i in range(0,len(selectedParDirPaths)):
            parFileName, parFilePath = getParFiles(selectedParDirPaths[i])
            parFilePaths = parFilePaths + [parFileName + parFilePath]
            for j in range(0,len(parFilePath)):
                with open(parFilePath[j]) as f:
                    lines = f.readlines()
                parameter_list = [x[:len(x)-1].split(" = ") for x in lines]
                parArgs.append(parFileName + [pars[1] for pars in parameter_list])
                if i == 0:
                    parNames = parNames + ['ParFile'] + [pars[0] for pars in parameter_list]
        parArgsZero = [args[0] for args in parArgs]
        delArgs = [tcz for tcz in tableContentZero if tcz not in parArgsZero]
        parArgs = [args for args in parArgs if args[0] not in tableContentZero]
        for item in tree.get_children():
            for delArg in delArgs:
                if tree.set(item,0) == delArg:
                    tree.delete(item)
        for par in parNames:
            tree.column(par,width=85, stretch=False)
            tree.heading(par,text=par)
        for args in parArgs:
            tree.insert("", 'end', values=args)
    else:
        for item in tree.get_children():
            tree.delete(item)

modelBtn = Button(LbModels_frame, text='Print Selected', command=selected_models)
modelBtn.grid(row=1, column=1, padx=10, pady=5)

for par in parNames:
    tree.column(par,width=85, stretch=False)
    tree.heading(par,text=par)

scrollbar = ttk.Scrollbar(Parameter_frame, orient=tkinter.HORIZONTAL,command=tree.xview)
scrollbar.grid(row=1,column=0,sticky='ew')
tree.configure(xscrollcommand=scrollbar.set)
tree.grid(column=0,row=0,sticky='nsew',padx=6,pady=6)

def select():
    global tree, parFilePaths, envDirName
    _, envName = os.path.split(envDirName)
    modelDirs = [path.dirname(path.dirname(parFilePath[1])) for parFilePath in parFilePaths]
    evallogPaths = [path.join(info[0], "eval_log",info[1][0][:-4]+".csv") for info in zip(modelDirs, parFilePaths)]
    trainlogPaths = [path.join(info[0], "train_log",info[1][0][:-4]+".csv") for info in zip(modelDirs, parFilePaths)]
    if envName == 'Linear1dPDEEnv-v0' or envName == 'Linear2dPDEEnv-v0':
        createWidgets(envName, 'Average Evaluation Return', 'steps', 'abs(Average Return - Average Exact Return)', evallogPaths, 'evalExactAndApproxDiff')
        createWidgets(envName, 'Average Evaluation Return', 'steps', 'log(Average Return/Average Exact Return)', evallogPaths, 'evalExactAndApproxLogQuot')
        createWidgets(envName, 'Episode Reward', 'steps', 'abs(Episode Reward)', trainlogPaths, 'trainAbs')
        createWidgets(envName, 'Episode Reward', 'steps', 'log(Episode Reward)', trainlogPaths, 'trainLogAbs')
    if envName == 'NonLinearPDEEnv-v0':
        createWidgets(envName, 'Average Evaluation Return', 'steps', 'abs(Average Return)', evallogPaths, 'evalAbs')
        createWidgets(envName, 'Average Evaluation Return', 'steps', 'log(Average Return)', evallogPaths, 'evalLogAbs')
        createWidgets(envName, 'Episode Reward', 'steps', 'abs(Episode Reward)', trainlogPaths, 'trainAbs')
        createWidgets(envName, 'Episode Reward', 'steps', 'log(Episode Reward)', trainlogPaths, 'trainLogAbs')

tree.bind("<Return>", lambda e: select())

def delete():
    global tree, parFilePaths
    tableContentZero = [tree.set(item,0) for item in tree.selection()]
    parFilePaths = [parFilePath for parFilePath in parFilePaths if parFilePath[0] not in tableContentZero]
    for item in tree.selection():
        tree.delete(item)

deleteBtn = ttk.Button(DeleteBtn_Frame, text= "Clear", command=delete)
deleteBtn.grid(row=0, column=1, padx=10, pady=5)

def deleteAll():
    global tree, parFilePaths
    parFilePaths = []
    for item in tree.get_children():
        tree.delete(item)

deleteAllBtn = ttk.Button(DeleteBtn_Frame, text= "Clear All", command=deleteAll)
deleteAllBtn.grid(row=0, column=2, padx=10, pady=5)

# Run app
app.mainloop()