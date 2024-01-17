import os
import os.path as osp
import tkinter
from tkinter import Listbox, Button, Frame, ttk, messagebox
import matplotlib.pyplot as plt
from  matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import pandas as pd
import numpy as np

def readSubDirNamesAndPaths(folderDir):
    subDirNames = []
    subDirPaths = []
    for file in os.listdir(folderDir):
        d = osp.join(folderDir, file)
        if osp.isdir(d):
            _, basename = osp.split(d)
            subDirNames.append(basename)
            subDirPaths.append(d)
    return subDirNames, subDirPaths

def getParFiles(parDirPaths):
    parFileNames = []
    parFilePaths = []
    for file in os.listdir(parDirPaths):
        d = osp.join(parDirPaths, file)
        _, basename = osp.split(d)
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
    return ax.plot(avgCostEval[:,0], avgCostEval[:,1])

def evallogPlotLog(ax, evallog_path):
    avgCostEval = pd.read_csv(evallog_path).values
    return ax.plot(avgCostEval[:,0], -np.log(np.abs(avgCostEval[:,1])))

def trainlogPlotAbs(ax, trainlog_path,label):
    epRewardTraining = pd.read_csv(trainlog_path).values
    return ax.plot(epRewardTraining[:,0], epRewardTraining[:,1])

def trainlogPlotLog(ax, trainlog_path,label):
    epRewardTraining = pd.read_csv(trainlog_path).values
    return ax.plot(epRewardTraining[:,0], -np.log(np.abs(epRewardTraining[:,1])))

def combinedtrainlogPlotAbs(ax, file_paths):
    combined = []
    for i in range(0,len(file_paths)):
        trainlogi = pd.read_csv(file_paths[i]).values.tolist()
        combined.extend(trainlogi)
    combinedArray = np.array(combined)
    dataset = pd.DataFrame({'Iteration': combinedArray[:, 0], 'Reward': combinedArray[:, 1]})
    return sns.lineplot(ax=ax, x = "Iteration", y = "Reward", data = dataset)

def combinedtrainlogPlotLog(ax, file_paths):
    combined = []
    for i in range(0,len(file_paths)):
        trainlogi = pd.read_csv(file_paths[i]).values.tolist()
        combined.extend(trainlogi)
    combinedArray = np.array(combined)
    dataset = pd.DataFrame({'Iteration': combinedArray[:, 0], 'Reward': -np.log(np.abs(combinedArray[:, 1]))})
    return sns.lineplot(ax=ax, x = "Iteration", y = "Reward", data = dataset)

def regularLinePlot(ax, x, y, dataset):
    return sns.lineplot(ax=ax, x = x, y = y, data = dataset)

def createWidgets(envName, title, xlabel, ylabel, filepaths, type, combineType):

    widget = tkinter.Tk()
    widget.title(title)
    widget.geometry("550x450")

    f0 = tkinter.Frame(widget)
    plt.rcParams.update({"text.usetex": True, "font.family": "monospace", "font.monospace": 'Computer Modern Typewriter'})
    fig = plt.figure(figsize=(8, 8))
   
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)

    if combineType=='combinedLog':
        if type == 'evalAbs':
            combinedtrainlogPlotAbs(ax1, filepaths)
        if type == 'evalLogAbs':
            combinedtrainlogPlotLog(ax1, filepaths)
        if type == 'trainAbs':
            combinedtrainlogPlotAbs(ax1, filepaths)
        if type == 'trainLogAbs':
            combinedtrainlogPlotLog(ax1, filepaths)
    if combineType=='singleLog':
        labels = ['$L = 5$', '$L = 10$']
        for i in range(0,len(filepaths)):
            if envName == 'Linear1dPDEEnv-v0' or envName == 'Linear2dPDEEnv-v0':
                if type == 'evalExactAndApproxDiff':
                    evallogPlotDiffApproxExact(ax1, filepaths[i])
                if type == 'evalExactAndApproxLogQuot':
                    evallogPlotLogQuotApproxExact(ax1, filepaths[i])
            if type == 'trainAbs':
                trainlogPlotAbs(ax1, filepaths[i], label=labels[i])
            if type == 'trainLogAbs':
                trainlogPlotLog(ax1, filepaths[i], label=labels[i])

    #plt.legend(fontsize = 14)
    canvas = FigureCanvasTkAgg(fig, f0)
    toolbar = NavigationToolbar2Tk(canvas, f0)
    toolbar.update()
    canvas._tkcanvas.pack(fill=tkinter.BOTH, expand=1)
    f0.pack(fill=tkinter.BOTH, expand=1)

def createWidgetsNormPlot(ylabel, data, colNames):

    widget = tkinter.Tk()
    widget.geometry("550x450")

    f0 = tkinter.Frame(widget)
    plt.rcParams.update({"text.usetex": True, "font.family": "monospace", "font.monospace": 'Computer Modern Typewriter'})
    fig = plt.figure(figsize=(8, 8))
   
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('$t$', fontsize=14)
    ax1.set_ylabel(ylabel[1], fontsize=14)

    regularLinePlot(ax1, colNames[0], colNames[1], data)

    canvas = FigureCanvasTkAgg(fig, f0)
    toolbar = NavigationToolbar2Tk(canvas, f0)
    toolbar.update()
    canvas._tkcanvas.pack(fill=tkinter.BOTH, expand=1)
    f0.pack(fill=tkinter.BOTH, expand=1)

def createWidgetsMultiNormPlot(ylabel, data, colNames,xlabel):

    widget = tkinter.Tk()
    widget.geometry("550x450")

    f0 = tkinter.Frame(widget)
    plt.rcParams.update({"text.usetex": True, "font.family": "monospace", "font.monospace": 'Computer Modern Typewriter'})
    fig = plt.figure(figsize=(8, 8))
   
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(' ', fontsize=14)

    for i in range(1,data.shape[1]):
      df = data.iloc[:,[0,i]]
      sns.lineplot(ax=ax1, x = colNames[0], y = colNames[i], data = df, label=ylabel[i])
    plt.legend(fontsize = 14)

    canvas = FigureCanvasTkAgg(fig, f0)
    toolbar = NavigationToolbar2Tk(canvas, f0)
    toolbar.update()
    canvas._tkcanvas.pack(fill=tkinter.BOTH, expand=1)
    f0.pack(fill=tkinter.BOTH, expand=1)

# App frames
app = tkinter.Tk()
app.title("ResultViewer")
app.geometry('800x500')
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

LbCheckpoints_frame = Frame(Lb_frame)
LbCheckpoints_frame.grid(row=0, column=3, padx=10, pady=5)

Parameter_frame = tkinter.LabelFrame(app ,text='Parameter')
Parameter_frame.columnconfigure(0, weight=1)
Parameter_frame.rowconfigure(0, weight=1)
Parameter_frame.grid(column=0,row=1,sticky='nsew',padx=6,pady=6)

DeleteBtn_Frame = Frame(Parameter_frame)
DeleteBtn_Frame.grid(row=2, column=0, padx=10, pady=5)

# UI elements

# Outputs folder
mainDir = osp.dirname(osp.dirname(__file__))
outputsDir = osp.join(mainDir, "outputs")

envNames, envFolderPaths = readSubDirNamesAndPaths(outputsDir)
LbEnvs = Listbox(LbEnvs_frame, selectmode=tkinter.SINGLE)
for i in range(0,len(envNames)):
    LbEnvs.insert(i, envNames[i])
LbEnvs.grid(row=0, column=0, padx=10, pady=5)

LbModels = Listbox(LbModels_frame, selectmode=tkinter.MULTIPLE, width = 25)
LbModels.grid(row=0, column=1, padx=10, pady=5)

LbCheckpoints = Listbox(LbCheckpoints_frame, selectmode=tkinter.SINGLE, width = 25)
LbCheckpoints.grid(row=0, column=0, padx=10, pady=5)

def createStateAndControlDatasets(envName, combinedArray):
    dataset = []
    yAxesLabels = []
    roundedTimes = np.round(combinedArray[:, 0],2)
    if envName == 'Linear1dPDEEnv-v0':
        yAxesLabels = ['$t$','$\\varepsilon_{{hM}}^{{rel}}(t)$','$\\varepsilon_{{hM}}(t)$','$\eta_y^h(t)$','$\eta_y^M(t)$',
                       '$\\left( \eta_u^{{hM, rel}}(t) \\right)_1$','$\\left( \eta_u^{{hM, rel}}(t) \\right)_2$','$\\left( \eta_u^{{hM, rel}}(t) \\right)_3$',
                       '$\\left( \eta_u^{{hM}}(t) \\right)_1$','$\\left( \eta_u^{{hM}}(t) \\right)_2$','$\\left( \eta_u^{{hM}}(t) \\right)_3$',
                       '$\\left( \eta_u^M(t) \\right)_1$','$\\left( \eta_u^M(t) \\right)_2$','$\\left( \eta_u^M(t) \\right)_3$',
                       '$\\left( \eta_u^h(t) \\right)_1$','$\\left( \eta_u^h(t) \\right)_2$','$\\left( \eta_u^h(t) \\right)_3$']
        dataset = pd.DataFrame({'t': roundedTimes, 'L2RelErrorStateExState': (combinedArray[:, 1]/(combinedArray[:, 2]+1e-16)), 'L2ErrorStateExState': combinedArray[:, 1], 'L2NormStateEx': combinedArray[:, 2], 'L2NormState': combinedArray[:, 3], 
                                'RelErroru1exu1': (combinedArray[:, 4]/(combinedArray[:, 10] + 1e-16)), 'RelErroru2exu2': (combinedArray[:, 5]/(combinedArray[:, 11] + 1e-16)), 'RelErroru3exu3': (combinedArray[:, 6]/(combinedArray[:, 12] + 1e-16)),
                                'Absu1exu1': combinedArray[:, 4], 'Absu2exu2': combinedArray[:, 5], 'Absu3exu3': combinedArray[:, 6],
                                'Absu1': combinedArray[:, 7], 'Absu2': combinedArray[:, 8], 'Absu3': combinedArray[:, 9], 
                                'Absu1ex': combinedArray[:, 10], 'Absu2ex': combinedArray[:, 11], 'Absu3ex': combinedArray[:, 12]})

    if envName == 'Linear2dPDEEnv-v0':
        yAxesLabels = ['$t$','$\\varepsilon_{{hM}}^{{rel}}(t)$','$\\varepsilon_{{hM}}(t)$','$\eta_y^h(t)$','$\eta_y^M(t)$',
                       '$\\left( \eta_u^{{hM, rel}}(t) \\right)_1$','$\\left( \eta_u^{{hM, rel}}(t) \\right)_2$','$\\left( \eta_u^{{hM, rel}}(t) \\right)_3$',
                       '$\\left( \eta_u^{{hM, rel}}(t) \\right)_4$','$\\left( \eta_u^{{hM, rel}}(t) \\right)_5$','$\\left( \eta_u^{{hM, rel}}(t) \\right)_6$',
                       '$\\left( \eta_u^{{hM, rel}}(t) \\right)_7$','$\\left( \eta_u^{{hM, rel}}(t) \\right)_8$','$\\left( \eta_u^{{hM, rel}}(t) \\right)_9$',
                       '$\\left( \eta_u^{{hM}}(t) \\right)_1$','$\\left( \eta_u^{{hM}}(t) \\right)_2$','$\\left( \eta_u^{{hM}}(t) \\right)_3$',
                       '$\\left( \eta_u^{{hM}}(t) \\right)_4$','$\\left( \eta_u^{{hM}}(t) \\right)_5$','$\\left( \eta_u^{{hM}}(t) \\right)_6$',
                       '$\\left( \eta_u^{{hM}}(t) \\right)_7$','$\\left( \eta_u^{{hM}}(t) \\right)_8$','$\\left( \eta_u^{{hM}}(t) \\right)_9$',
                       '$\\left( \eta_u^M(t) \\right)_1$','$\\left( \eta_u^M(t) \\right)_2$','$\\left( \eta_u^M(t) \\right)_3$',
                       '$\\left( \eta_u^M(t) \\right)_4$','$\\left( \eta_u^M(t) \\right)_5$','$\\left( \eta_u^M(t) \\right)_6$',
                       '$\\left( \eta_u^M(t) \\right)_7$','$\\left( \eta_u^M(t) \\right)_8$','$\\left( \eta_u^M(t) \\right)_9$',
                       '$\\left( \eta_u^h(t) \\right)_1$','$\\left( \eta_u^h(t) \\right)_2$','$\\left( \eta_u^h(t) \\right)_3$',
                       '$\\left( \eta_u^h(t) \\right)_4$','$\\left( \eta_u^h(t) \\right)_5$','$\\left( \eta_u^h(t) \\right)_6$',
                       '$\\left( \eta_u^h(t) \\right)_7$','$\\left( \eta_u^h(t) \\right)_8$','$\\left( \eta_u^h(t) \\right)_9$']
        dataset = pd.DataFrame({'t': roundedTimes, 'L2RelErrorStateExState': (combinedArray[:, 1]/(combinedArray[:, 2]+1e-16)), 'L2ErrorStateExState': combinedArray[:, 1], 'L2NormStateEx': combinedArray[:, 2], 'L2NormState': combinedArray[:, 3], 
                                'RelErroru1exu1': (combinedArray[:, 4]/(combinedArray[:, 22]+1e-16)), 'RelErroru2exu2': (combinedArray[:, 5]/(combinedArray[:, 23]+1e-16)), 'RelErroru3exu3': (combinedArray[:, 6]/(combinedArray[:, 24]+1e-16)),
                                'RelErroru4exu4': (combinedArray[:, 7]/(combinedArray[:, 25]+1e-16)), 'RelErroru5exu5': (combinedArray[:, 8]/(combinedArray[:, 26]+1e-16)), 'RelErroru6exu6': (combinedArray[:, 9]/(combinedArray[:, 27]+1e-16)),
                                'RelErroru7exu7': (combinedArray[:, 10]/(combinedArray[:, 28]+1e-16)), 'RelErroru8exu8': (combinedArray[:, 11]/(combinedArray[:, 29]+1e-16)), 'RelErroru9exu9': (combinedArray[:, 12]/(combinedArray[:, 30]+1e-16)),
                                'Absu1exu1': combinedArray[:, 4], 'Absu2exu2': combinedArray[:, 5], 'Absu3exu3': combinedArray[:, 6],
                                'Absu4exu4': combinedArray[:, 7], 'Absu5exu5': combinedArray[:, 8], 'Absu6exu6': combinedArray[:, 9],
                                'Absu7exu7': combinedArray[:, 10], 'Absu8exu8': combinedArray[:, 11], 'Absu9exu9': combinedArray[:, 12],
                                'Absu1': combinedArray[:, 13], 'Absu2': combinedArray[:, 14], 'Absu3': combinedArray[:, 15],
                                'Absu4': combinedArray[:, 16], 'Absu5': combinedArray[:, 17], 'Absu6': combinedArray[:, 18],
                                'Absu7': combinedArray[:, 19], 'Absu8': combinedArray[:, 20], 'Absu9': combinedArray[:, 21],                               
                                'Absu1ex': combinedArray[:, 22], 'Absu2ex': combinedArray[:, 23], 'Absu3ex': combinedArray[:, 24],
                                'Absu4ex': combinedArray[:, 25], 'Absu5ex': combinedArray[:, 26], 'Absu6ex': combinedArray[:, 27],
                                'Absu7ex': combinedArray[:, 28], 'Absu8ex': combinedArray[:, 29], 'Absu9ex': combinedArray[:, 30]})

    if envName == 'NonLinearPDEEnv-v0':
        yAxesLabels = ['$t$','$\eta_y^M(t)$','$\\left( \eta_u^M(t) \\right)_1$','$\\left( \eta_u^M(t) \\right)_2$','$\\left( \eta_u^M(t) \\right)_3$']
        dataset = pd.DataFrame({'t': roundedTimes, 'L2NormState': combinedArray[:, 1], 'Absu1': combinedArray[:, 2], 'Absu2': combinedArray[:, 3], 'Absu3': combinedArray[:, 4]})

    return yAxesLabels, dataset    

def combinedNorm():
    global normFilePaths, envDirName
    _, envName = osp.split(envDirName)

    combined = []
    for i in range(0,len(normFilePaths)):
        normFilei = pd.read_csv(normFilePaths[i], header=None, index_col=0).values.tolist()
        combined.extend(normFilei[1:])
    combinedArray = np.array(combined, dtype=float)

    yAxesLabels, dataset = createStateAndControlDatasets(envName, combinedArray)
    
    if envName=='Linear1dPDEEnv-v0':
        df1 = dataset[['t', 'L2RelErrorStateExState']]
        df1['L2RelErrorStateExState'] = np.log(df1['L2RelErrorStateExState'])
        colNames1 = df1.columns
        yLabels1 = [yAxesLabels[0], yAxesLabels[1]]
        createWidgetsNormPlot(yLabels1, df1, colNames1)
        df2 = dataset[['t', 'L2NormState', 'L2NormStateEx']]
        colNames2 = df2.columns
        yLabels2 = [yAxesLabels[0], yAxesLabels[4], yAxesLabels[3]]
        createWidgetsMultiNormPlot(yLabels2, df2, colNames2,'$t$')
        df3 = dataset[['t', 'Absu1', 'Absu1ex']]
        df4 = dataset[['t', 'Absu2', 'Absu2ex']]
        df5 = dataset[['t', 'Absu3', 'Absu3ex']]
        colNames3 = df3.columns
        colNames4 = df4.columns
        colNames5 = df5.columns
        yLabels3 = [yAxesLabels[0], yAxesLabels[11], yAxesLabels[14]]
        yLabels4 = [yAxesLabels[0], yAxesLabels[12], yAxesLabels[15]]
        yLabels5 = [yAxesLabels[0], yAxesLabels[13], yAxesLabels[16]]
        createWidgetsMultiNormPlot(yLabels3, df3, colNames3,'$t$')
        createWidgetsMultiNormPlot(yLabels4, df4, colNames4,'$t$')
        createWidgetsMultiNormPlot(yLabels5, df5, colNames5,'$t$')
        df6 = dataset[['t', 'RelErroru1exu1']]
        df6['RelErroru1exu1'] = np.log(df6['RelErroru1exu1'])
        df7 = dataset[['t', 'RelErroru2exu2']]
        df7['RelErroru2exu2'] = np.log(df7['RelErroru2exu2'])
        df8 = dataset[['t', 'RelErroru3exu3']]
        df8['RelErroru3exu3'] = np.log(df8['RelErroru3exu3'])
        colNames6 = df6.columns
        colNames7 = df7.columns
        colNames8 = df8.columns
        yLabels6 = [yAxesLabels[0], yAxesLabels[5]]
        yLabels7 = [yAxesLabels[0], yAxesLabels[6]]
        yLabels8 = [yAxesLabels[0], yAxesLabels[7]]
        createWidgetsMultiNormPlot(yLabels6, df6, colNames6,'$t$')
        createWidgetsMultiNormPlot(yLabels7, df7, colNames7,'$t$')
        createWidgetsMultiNormPlot(yLabels8, df8, colNames8,'$t$')
    if envName=='Linear2dPDEEnv-v0':
        df1 = dataset[['t', 'L2RelErrorStateExState']]
        df1['L2RelErrorStateExState'] = np.log(df1['L2RelErrorStateExState'])
        colNames1 = df1.columns
        yLabels1 = [yAxesLabels[0], yAxesLabels[1]]
        createWidgetsNormPlot(yLabels1, df1, colNames1)
        df2 = dataset[['t', 'L2NormState', 'L2NormStateEx']]
        colNames2 = df2.columns
        yLabels2 = [yAxesLabels[0], yAxesLabels[4], yAxesLabels[3]]
        createWidgetsMultiNormPlot(yLabels2, df2, colNames2,'$t$')
        df3 = dataset[['t', 'Absu1', 'Absu1ex']]
        df4 = dataset[['t', 'Absu2', 'Absu2ex']]
        df5 = dataset[['t', 'Absu3', 'Absu3ex']]
        df6 = dataset[['t', 'Absu4', 'Absu4ex']]
        df7 = dataset[['t', 'Absu5', 'Absu5ex']]
        df8 = dataset[['t', 'Absu6', 'Absu6ex']]
        df9 = dataset[['t', 'Absu7', 'Absu7ex']]
        df10 = dataset[['t', 'Absu8', 'Absu8ex']]
        df11 = dataset[['t', 'Absu9', 'Absu9ex']]
        colNames3 = df3.columns
        colNames4 = df4.columns
        colNames5 = df5.columns
        colNames6 = df6.columns
        colNames7 = df7.columns
        colNames8 = df8.columns
        colNames9 = df9.columns
        colNames10 = df10.columns
        colNames11 = df11.columns
        yLabels3 = [yAxesLabels[0], yAxesLabels[23], yAxesLabels[32]]
        yLabels4 = [yAxesLabels[0], yAxesLabels[24], yAxesLabels[33]]
        yLabels5 = [yAxesLabels[0], yAxesLabels[25], yAxesLabels[34]]
        yLabels6 = [yAxesLabels[0], yAxesLabels[26], yAxesLabels[35]]
        yLabels7 = [yAxesLabels[0], yAxesLabels[27], yAxesLabels[36]]
        yLabels8 = [yAxesLabels[0], yAxesLabels[28], yAxesLabels[37]]
        yLabels9 = [yAxesLabels[0], yAxesLabels[29], yAxesLabels[38]]
        yLabels10 = [yAxesLabels[0], yAxesLabels[30], yAxesLabels[39]]
        yLabels11 = [yAxesLabels[0], yAxesLabels[31], yAxesLabels[40]]
        createWidgetsMultiNormPlot(yLabels3, df3, colNames3,'$t$')
        createWidgetsMultiNormPlot(yLabels4, df4, colNames4,'$t$')
        createWidgetsMultiNormPlot(yLabels5, df5, colNames5,'$t$')
        createWidgetsMultiNormPlot(yLabels6, df6, colNames6,'$t$')
        createWidgetsMultiNormPlot(yLabels7, df7, colNames7,'$t$')
        createWidgetsMultiNormPlot(yLabels8, df8, colNames8,'$t$')
        createWidgetsMultiNormPlot(yLabels9, df9, colNames9,'$t$')
        createWidgetsMultiNormPlot(yLabels10, df10, colNames10,'$t$')
        createWidgetsMultiNormPlot(yLabels11, df11, colNames11,'$t$')
        df12 = dataset[['t', 'RelErroru1exu1']]
        df12['RelErroru1exu1'] = np.log(df12['RelErroru1exu1'])
        df13 = dataset[['t', 'RelErroru2exu2']]
        df13['RelErroru2exu2'] = np.log(df13['RelErroru2exu2'])
        df14 = dataset[['t', 'RelErroru3exu3']]
        df14['RelErroru3exu3'] = np.log(df14['RelErroru3exu3'])
        df15 = dataset[['t', 'RelErroru4exu4']]
        df15['RelErroru4exu4'] = np.log(df15['RelErroru4exu4'])
        df16 = dataset[['t', 'RelErroru5exu5']]
        df16['RelErroru5exu5'] = np.log(df16['RelErroru5exu5'])
        df17 = dataset[['t', 'RelErroru6exu6']]
        df17['RelErroru6exu6'] = np.log(df17['RelErroru6exu6'])
        df18 = dataset[['t', 'RelErroru7exu7']]
        df18['RelErroru7exu7'] = np.log(df18['RelErroru7exu7'])
        df19 = dataset[['t', 'RelErroru8exu8']]
        df19['RelErroru8exu8'] = np.log(df19['RelErroru8exu8'])
        df20 = dataset[['t', 'RelErroru9exu9']]
        df20['RelErroru9exu9'] = np.log(df20['RelErroru9exu9'])
        colNames12 = df12.columns
        colNames13 = df13.columns
        colNames14 = df14.columns
        colNames15 = df15.columns
        colNames16 = df16.columns
        colNames17 = df17.columns
        colNames18 = df18.columns
        colNames19 = df19.columns
        colNames20 = df20.columns
        yLabels12 = [yAxesLabels[0], yAxesLabels[5]]
        yLabels13 = [yAxesLabels[0], yAxesLabels[6]]
        yLabels14 = [yAxesLabels[0], yAxesLabels[7]]
        yLabels15 = [yAxesLabels[0], yAxesLabels[8]]
        yLabels16 = [yAxesLabels[0], yAxesLabels[9]]
        yLabels17 = [yAxesLabels[0], yAxesLabels[10]]
        yLabels18 = [yAxesLabels[0], yAxesLabels[11]]
        yLabels19 = [yAxesLabels[0], yAxesLabels[12]]
        yLabels20 = [yAxesLabels[0], yAxesLabels[13]]
        createWidgetsMultiNormPlot(yLabels12, df12, colNames12,'$t$')
        createWidgetsMultiNormPlot(yLabels13, df13, colNames13,'$t$')
        createWidgetsMultiNormPlot(yLabels14, df14, colNames14,'$t$')
        createWidgetsMultiNormPlot(yLabels15, df15, colNames15,'$t$')
        createWidgetsMultiNormPlot(yLabels16, df16, colNames16,'$t$')
        createWidgetsMultiNormPlot(yLabels17, df17, colNames17,'$t$')
        createWidgetsMultiNormPlot(yLabels18, df18, colNames18,'$t$')
        createWidgetsMultiNormPlot(yLabels19, df19, colNames19,'$t$')
        createWidgetsMultiNormPlot(yLabels20, df20, colNames20,'$t$')
    if envName == 'NonLinearPDEEnv-v0':
        df1 = dataset[['t', 'L2NormState']]
        colNames1 = df1.columns
        yLabels1 = [yAxesLabels[0], yAxesLabels[1]]
        createWidgetsNormPlot(yLabels1, df1, colNames1)
        df2 = dataset[['t', 'Absu1']]
        df3 = dataset[['t', 'Absu2']]
        df4 = dataset[['t', 'Absu3']]
        colNames2 = df2.columns
        colNames3 = df3.columns
        colNames4 = df4.columns
        yLabels2 = [yAxesLabels[0], yAxesLabels[2]]
        yLabels3 = [yAxesLabels[0], yAxesLabels[3]]
        yLabels4 = [yAxesLabels[0], yAxesLabels[4]]
        createWidgetsNormPlot(yLabels2, df2, colNames2)
        createWidgetsNormPlot(yLabels3, df3, colNames3)
        createWidgetsNormPlot(yLabels4, df4, colNames4)

def singleNorm():
    
    print('tbd')

    #global normFilePaths, envDirName
    #cursor = LbCheckpoints.curselection()
    #_, envName = osp.split(envDirName)

    #normFile = pd.read_csv(normFilePaths[cursor[0]], header=None, index_col=0).values.tolist()
    #combined = []
    #combined.extend(normFile[1:])
    #combinedArray = np.array(combined, dtype=float)

    #yAxesLabels, dataset = createStateAndControlDatasets(envName, combinedArray)

    #for i in range(1,dataset.shape[1]):
    #  df = dataset.iloc[:,[0,i]]
    #  colNames = df.columns
    #  createWidgetsNormPlot(yAxesLabels[0], yAxesLabels[i], df, colNames[0], colNames[1])

combinedNormsPlotBtn = Button(LbCheckpoints_frame, text='Combined Norm Plot', command=combinedNorm)
combinedNormsPlotBtn.grid(row=1, column=0, padx=10, pady=5)

singleNormsPlotBtn = Button(LbCheckpoints_frame, text='Single Norm Plot', command=singleNorm)
singleNormsPlotBtn.grid(row=2, column=0, padx=10, pady=5)

envDirName = []
parFilePaths = []
parNames = ['ParFile', 'dimS', 'dimA', 'h', 'ctrl_range', 'env_id', 'loadModel', 'resumeTraining', 'verboseLoopTraining', 'useExactSolution', 'useMatCARE', 'Kpath', 'model', 'L', 'T', 'time_steps', 'gamma', 'lr', 'sigma', 'polyak', 'max_iter', 'num_checkpoints', 'buffer_size', 'fill_buffer', 'batch_size', 'train_interval', 'start_train', 'eval_interval', 'smooth', 'double', 'noise', 'h_scale', 'device', 'render']
tree = ttk.Treeview(Parameter_frame, columns=parNames, show='headings', height=5, selectmode='extended')

def selected_env():
    global LbEnvs, LbModels, exactModelNames, envDirName, envFolderPaths, tree
    for item in tree.get_children():
            tree.delete(item)
    cursor =  LbEnvs.curselection()
    envDirName = []
    exactModelNames = []
    if len(cursor)!=0:
        envDir = envFolderPaths[cursor[0]]
        envDirName = envDir
        modelNames, _ = readSubDirNamesAndPaths(envDir)
        dates = [models.split("_")[1].split("T")[0] for models in modelNames]
        dates = [date[8:10]+"."+date[5:7]+"."+date[2:4] for date in dates]
        times = [models.split("_")[1].split("T")[1] for models in modelNames]
        times = [time[0:2]+":"+time[2:4]+":"+time[4:6] for time in times]
        listitem = ["HJDQN "+info[0]+" "+info[1] for info in zip(dates, times)]
        LbModels.delete(0,tkinter.END)
        for i in range(0,len(listitem)):
            LbModels.insert(i, listitem[i])
            exactModelNames.append(modelNames[i])
 
envBtn = Button(LbEnvs_frame, text='Print Selected', command=selected_env)
envBtn.grid(row=1, column=0, padx=10, pady=5)

def selected_models():
    global LbModels, envDirName, parFilePaths, tree, exactModelNames, normFilePaths
    cursor =  LbModels.curselection()

    selectedModel = exactModelNames[cursor[0]]
    normsFolder = osp.join(osp.dirname(osp.dirname(__file__)), "outputs", envDirName, selectedModel, "state_and_control_norms")
    normFilePaths = []
    normFileNames = []
    for file in os.listdir(normsFolder):
      normFileNames.append(file)
      normFilePaths.append(osp.join(normsFolder,file))
    LbCheckpoints.delete(0,tkinter.END)
    for i in range(0,len(normFileNames)):
        LbCheckpoints.insert(i, normFileNames[i])

    tableContentZero = [tree.set(item,0) for item in tree.get_children()]
    parFilePaths = []
    parNames = []
    parArgs = []
    if len(cursor)!=0:
        _, parDirPaths = readSubDirNamesAndPaths(envDirName)
        parDirPaths = [osp.join(modelPath,"parameter") for modelPath in parDirPaths]
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
    _, envName = osp.split(envDirName)
    modelDirs = [osp.dirname(osp.dirname(parFilePath[1])) for parFilePath in parFilePaths]
    evallogPaths = [osp.join(info[0], "eval_log",info[1][0][:-4]+".csv") for info in zip(modelDirs, parFilePaths)]
    trainlogPaths = [osp.join(info[0], "train_log",info[1][0][:-4]+".csv") for info in zip(modelDirs, parFilePaths)]
    if len(evallogPaths) < 3 and len(trainlogPaths) < 3: 
        if envName == 'Linear1dPDEEnv-v0' or envName == 'Linear2dPDEEnv-v0':
            createWidgets(envName, 'Evaluation Return', '$steps$', '$\\vert \mathcal{{R}}_h - \mathcal{{R}}_M \\vert$', evallogPaths, 'evalExactAndApproxDiff','singleLog')
            createWidgets(envName, 'Evaluation Return', '$steps$', '$\log \\left( \mathcal{{R}}_h \\right) - \log \\left( \mathcal{{R}}_M \\right)$', evallogPaths, 'evalExactAndApproxLogQuot','singleLog')
            createWidgets(envName, 'Episode Reward', '$steps$', '$\mathcal{{R}}_T$', trainlogPaths, 'trainAbs','singleLog')
            createWidgets(envName, 'Episode Reward', '$steps$', '$- \log \\left( \\vert \mathcal{{R}}_T \\vert \\right)$', trainlogPaths, 'trainLogAbs','singleLog')
        if envName == 'NonLinearPDEEnv-v0':
            createWidgets(envName, 'Episode Reward', '$steps$', '$\mathcal{{R}}_T$', trainlogPaths, 'trainAbs','singleLog')
            createWidgets(envName, 'Episode Reward', '$steps$', '$- \log \\left( \\vert \mathcal{{R}}_T \\vert \\right)$', trainlogPaths, 'trainLogAbs','singleLog')
    else:
        if envName == 'Linear1dPDEEnv-v0' or envName == 'Linear2dPDEEnv-v0':
            createWidgets(envName, 'Episode Reward', '$steps$', '$\mathcal{{R}}_T$', trainlogPaths, 'trainAbs','combinedLog')
            createWidgets(envName, 'Episode Reward', '$steps$', '$- \log \\left( \\vert \mathcal{{R}}_T \\vert \\right)$', trainlogPaths, 'trainLogAbs','combinedLog')
        if envName == 'NonLinearPDEEnv-v0':
            createWidgets(envName, 'Episode Reward', '$steps$', '$\mathcal{{R}}_T$', trainlogPaths, 'trainAbs','combinedLog')
            createWidgets(envName, 'Episode Reward', '$steps$', '$- \log \\left( \\vert \mathcal{{R}}_T \\vert \\right)$', trainlogPaths, 'trainLogAbs','combinedLog')

tree.bind("<Return>", lambda e: select())

def delete():
    global tree, parFilePaths
    tableContentZero = [tree.set(item,0) for item in tree.selection()]
    parFilePaths = [parFilePath for parFilePath in parFilePaths if parFilePath[0] not in tableContentZero]
    for item in tree.selection():
        tree.delete(item)

deleteBtn = ttk.Button(DeleteBtn_Frame, text= "Clear", command=delete)
deleteBtn.grid(row=0, column=2, padx=10, pady=5)

def deleteAll():
    global tree, parFilePaths
    parFilePaths = []
    for item in tree.get_children():
        tree.delete(item)

deleteAllBtn = ttk.Button(DeleteBtn_Frame, text= "Clear All", command=deleteAll)
deleteAllBtn.grid(row=0, column=3, padx=10, pady=5)

# Run app
app.mainloop()