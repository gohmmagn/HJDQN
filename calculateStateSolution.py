import numpy as np
import os
import time
import sys
from os import path
import json
import pandas as pd
import gymnasium as gym
from algorithms.hjdqn.hjdqn_agent import HJDQNAgent
from algorithms.ddpg.ddpg_agent import DDPGAgent
from algorithms.utils import set_log_dir, get_env_spec, scaled_env
from algorithms.noise import IndependentGaussian, Zero, SDE
import gym_lqr
import dolfinx
import ufl
from mpi4py import MPI
import petsc4py
petsc4py.init()
from petsc4py import PETSc
from dolfinx import mesh, fem, io, nls
from dolfinx.fem import petsc as femPetsc
from dolfinx.nls import petsc as nlsPetsc
from numpy.linalg import norm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algo', required=False, choices=['hjdqn', 'ddpg'], default='hjdqn')
parser.add_argument('--envId', required=True)
parser.add_argument('--device', required=False, default='cpu')
parser.add_argument('--modelName', required=False, default='all')
parser.add_argument('--savedModel', required=False, default='all')
parser.add_argument('--num_steps_l1', required=False, default=200, type=int)
parser.add_argument('--T_l1', required=False, default=2, type=float)
parser.add_argument('--num_steps_l2', required=False, default=500, type=int)
parser.add_argument('--T_l2', required=False, default=5, type=float)
parser.add_argument('--num_steps_nl', required=False, default=400, type=int)
parser.add_argument('--T_nl', required=False, default=4, type=float)

args = parser.parse_args()
algo = args.algo

def pdeSolutions_directory_path(envId, modelName):
  return "outputs/{}/{}/pde_solutions".format(envId, modelName)

def stateAndControlNorms_directory_path(envId, modelName):
  return "outputs/{}/{}/state_and_control_norms".format(envId, modelName)

def parameter_directory_path(envId, modelName):
  return "outputs/{}/{}/parameter".format(envId, modelName)

def checkpoints_diretory_path(envId, modelName):
  return "outputs/{}/{}/checkpoints".format(envId, modelName)

def checkpoints_fileDirectory(checkpointsDiretoryPath):
  return path.join(checkpointsDiretoryPath,'fileDirectory.json')

modelNames = []
for modelFolder in os.listdir("outputs/{}".format(args.envId)):
  modelNames.append(modelFolder)

singleOutput = 0
if args.modelName!='all' and args.savedModel!='all':
  modelNames = [args.modelName]
  singleOutput = 1

# Define PDE problem components for 1d-----------------------------------------------------------------------------------------

# Define initial time.
t1d = 0

# Time intervall.
dt1d = (args.T_l1/args.num_steps_l1)

# Define space interval.

#1d
Omega1d = [-1,1]

# Define mesh.

# One dimensional mesh.
nx1d = 200
domain1d = mesh.create_interval(MPI.COMM_WORLD, nx1d, Omega1d)

# Define test function space.
V1d = fem.functionspace(domain1d, ("P", 1))

# Define Dirichlet Boundary Condition. 
fdim1d = domain1d.topology.dim - 1
boundary_facets1d = mesh.locate_entities_boundary(domain1d, fdim1d, lambda x: np.full(x.shape[1], True, dtype=bool))
bc1d = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V1d, fdim1d, boundary_facets1d), V1d)

class Initial_condition1d():

    def __init__(self):
        pass

    def __call__(self,x):
        values = np.zeros(x.shape,dtype=PETSc.ScalarType)
        values = (x[0]-1)*(x[0]+1)+5
        return values

initial_condition1d = Initial_condition1d()

y_01d = fem.Function(V1d)
y_01d.name = "y_0"
y_01d.interpolate(initial_condition1d)

y_01dex = fem.Function(V1d)
y_01dex.name = "y_0"
y_01dex.interpolate(initial_condition1d)

# Trial and test function.
y1d, phi1d = ufl.TrialFunction(V1d), ufl.TestFunction(V1d)

# PDE coefficients
nu1d = 0.01

class PDE_a1d():

    def __init__(self):
        pass

    def __call__(self,x):
        values = np.zeros(x.shape,dtype=PETSc.ScalarType)
        values = np.sin(x[0]) #np.ones(x[0].shape), np.sin(x[0])
        return values

class PDE_b1d():

    def __init__(self):
        pass

    def __call__(self,x):
        values = np.zeros(x.shape,dtype=PETSc.ScalarType)
        values = -x[0] #np.ones(x[0].shape), -x[0]
        return values

a_V1d = fem.Function(V1d)
a_V1d.interpolate(PDE_a1d())

b_V11d = fem.Function(V1d)
b_V11d.interpolate(PDE_b1d())
b_V1d = ufl.as_vector([b_V11d])

# Indicator function for 1d PDE also called acctuator functions.
w11d = np.array([-0.7,-0.4])
w21d = np.array([-0.2,0.2])
w31d = np.array([0.4,0.7])
tol1d = 1e-14

acctuatorDomains1d = [w11d, w21d, w31d]

I_w1d = [fem.Function(V1d) for i in range(0,len(acctuatorDomains1d))]

for i, acct in enumerate(acctuatorDomains1d):
  I_w1d[i].interpolate(lambda x: np.where(np.logical_and(x[0] - acct[0] +tol1d >= 0, acct[1] + tol1d - x[0] >= 0), 1, 0))
    
acctuator1d = np.array([I_w1d[0].x.array])
for i in range(1,len(I_w1d)):
  acctuator1d = np.r_[acctuator1d, [I_w1d[i].x.array]]

resortIndex1d = "empty"

# Define solution variable, and interpolate initial condition.
y_n1d = fem.Function(V1d)
y_n1d.name = "y_n"
y_n1d.interpolate(initial_condition1d)
y_n1dex = fem.Function(V1d)
y_n1dex.name = "y_n"
y_n1dex.interpolate(initial_condition1d)

# Control vector components.
us1d = [fem.Constant(domain1d, 0.0) for i in range(0,len(I_w1d))]
us1dex = [fem.Constant(domain1d, 0.0) for i in range(0,len(I_w1d))]

# Variational Problem (LHS).
a_h1d = y1d*phi1d*ufl.dx + nu1d*dt1d*ufl.dot(ufl.grad(y1d),ufl.grad(phi1d))*ufl.dx + dt1d*a_V1d*y1d*phi1d*ufl.dx - dt1d*ufl.dot(b_V1d*y1d,ufl.grad(phi1d))*ufl.dx
bilinear_form1d = fem.form(a_h1d)

# Variational Problem (RHS).
L1d = us1d[0]*I_w1d[0]
for i in range(1,len(us1d)):
  L1d = L1d + us1d[i]*I_w1d[i]
L1d = (y_01d + dt1d*L1d)*phi1d*ufl.dx
linear_form1d = fem.form(L1d)

L1dex = us1dex[0]*I_w1d[0]
for i in range(1,len(us1dex)):
  L1dex = L1dex + us1dex[i]*I_w1d[i]
L1dex = (y_01dex + dt1d*L1dex)*phi1d*ufl.dx
linear_form1dex = fem.form(L1dex)

# Assemble FEM matrices.
C1d = fem.petsc.assemble_matrix(bilinear_form1d)
C1d.assemble()
d1d = fem.petsc.create_vector(linear_form1d)
d1dex = fem.petsc.create_vector(linear_form1dex)

# Create solver.
solver1d = PETSc.KSP().create(domain1d.comm)
solver1d.setOperators(C1d)
solver1d.setType(PETSc.KSP.Type.PREONLY)
solver1d.getPC().setType(PETSc.PC.Type.LU)

#Error function
e_W1d = fem.Function(V1d)
error_yn_ex_yn1d = fem.form(ufl.inner(e_W1d, e_W1d) * ufl.dx)
norm_yn_ex1d = fem.form(ufl.inner(y_n1dex, y_n1dex) * ufl.dx)
norm_yn1d = fem.form(ufl.inner(y_n1d, y_n1d) * ufl.dx)

#------------------------------------------------------------------------------------------------------------------------------
# Define PDE problem components for 2d-----------------------------------------------------------------------------------------

# Define initial time.
t2d = 0

# Time intervall.
dt2d = (args.T_l2/args.num_steps_l2)

# Define space interval.

# 2d
Omega2d = [np.array([0,0]), np.array([1,1])]

# Define mesh.

# Two dimensional mesh.
nx2d = 32
ny2d = 32
domain2d = mesh.create_unit_square(MPI.COMM_WORLD, nx2d, ny2d)

# Define test function space.
V2d = fem.functionspace(domain2d, ("P", 1))

# Define Dirichlet Boundary Condition. 
fdim2d = domain2d.topology.dim - 1
boundary_facets2d = mesh.locate_entities_boundary(domain2d, fdim2d, lambda x: np.full(x.shape[1], True, dtype=bool))
bc2d = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V2d, fdim2d, boundary_facets2d), V2d)

# Resorting index to reshape 1d output vectors.
mesh_coords = np.transpose(domain2d.geometry.x[:,0:2])
n = int(np.sqrt(mesh_coords.shape[1]))
IY = np.argsort(mesh_coords[1,:])
Y = mesh_coords[1,IY]
X = mesh_coords[0,IY]
for i in range(0,n):
  Xi = X[i*n:(i+1)*n]
  IYi = IY[i*n:(i+1)*n]
  IXi = np.argsort(Xi)
  Xi_s = Xi[IXi]
  X[i*n:(i+1)*n] = Xi_s
  IY[i*n:(i+1)*n] = IYi[IXi]
resortIndex2d = IY

class Initial_condition2d():

    def __init__(self):
      pass

    def __call__(self,x):
       values = np.zeros(x.shape,dtype=PETSc.ScalarType)
       values = 3*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
       return values

initial_condition2d = Initial_condition2d()

y_02d = fem.Function(V2d)
y_02d.name = "y_0"
y_02d.interpolate(initial_condition2d)

y_02dex = fem.Function(V2d)
y_02dex.name = "y_0"
y_02dex.interpolate(initial_condition2d)

# Trial and test function.
y2d, phi2d = ufl.TrialFunction(V2d), ufl.TestFunction(V2d)

# PDE coefficients
nu2d = 0.01

class PDE_a2d():

    def __init__(self):
      pass

    def __call__(self,x):
      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
      values = np.sin(x[0])
      return values

class PDE_b_12d():

    def __init__(self):
      pass

    def __call__(self,x):
      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
      values = -x[0]
      return values

class PDE_b_22d():

    def __init__(self):
      pass

    def __call__(self,x):
      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
      values = -x[1]
      return values

a_V2d = fem.Function(V2d)
a_V2d.interpolate(PDE_a2d())

b_V12d = fem.Function(V2d)
b_V22d = fem.Function(V2d)
b_V12d.interpolate(PDE_b_12d())
b_V22d.interpolate(PDE_b_22d())
b_V2d = ufl.as_vector([b_V12d, b_V22d])

# Indicator function for 2d PDE also called acctuator functions.
w12d = np.array([[0.1,0.3],[0.1,0.3]])
w22d = np.array([[0.4,0.6],[0.1,0.3]])
w32d = np.array([[0.7,0.9],[0.1,0.3]])
w42d = np.array([[0.1,0.3],[0.4,0.6]])
w52d = np.array([[0.4,0.6],[0.4,0.6]])
w62d = np.array([[0.7,0.9],[0.4,0.6]])
w72d = np.array([[0.1,0.3],[0.7,0.9]])
w82d = np.array([[0.4,0.6],[0.7,0.9]])
w92d = np.array([[0.7,0.9],[0.7,0.9]])
tol1d = 1e-14

acctuatorDomains2d = [w12d, w22d, w32d, w42d, w52d, w62d, w72d, w82d, w92d]

I_w2d = [fem.Function(V2d) for i in range(0,len(acctuatorDomains2d))]

for i, acct in enumerate(acctuatorDomains2d):
  I_w2d[i].interpolate(lambda x: np.where(np.logical_and(np.logical_and(x[0]>=acct[0,0], x[0]<=acct[0,1]),np.logical_and(x[1]>=acct[1,0], x[1]<=acct[1,1])), 1, 0))
    
acctuator2d = np.array([I_w2d[0].x.array])
for i in range(1,len(I_w2d)):
  acctuator2d = np.r_[acctuator2d, [I_w2d[i].x.array]]

# Define solution variable, and interpolate initial condition.
y_n2d = fem.Function(V2d)
y_n2d.name = "y_n"
y_n2d.interpolate(initial_condition2d)
y_n2dex = fem.Function(V2d)
y_n2dex.name = "y_n"
y_n2dex.interpolate(initial_condition2d)

# Control vector components.
us2d = [fem.Constant(domain2d, 0.0) for i in range(0,len(I_w2d))]
us2dex = [fem.Constant(domain2d, 0.0) for i in range(0,len(I_w2d))]

# Variational Problem (LHS).
a_h2d = y2d*phi2d*ufl.dx + nu2d*dt2d*ufl.dot(ufl.grad(y2d),ufl.grad(phi2d))*ufl.dx + dt2d*a_V2d*y2d*phi2d*ufl.dx - dt2d*ufl.dot(b_V2d*y2d,ufl.grad(phi2d))*ufl.dx
bilinear_form2d = fem.form(a_h2d)

# Variational Problem (RHS).
L2d = us2d[0]*I_w2d[0]
for i in range(1,len(us2d)):
  L2d = L2d + us2d[i]*I_w2d[i]
L2d = (y_02d + dt2d*L2d)*phi2d*ufl.dx
linear_form2d = fem.form(L2d)

L2dex = us2dex[0]*I_w2d[0]
for i in range(1,len(us2dex)):
  L2dex = L2dex + us2dex[i]*I_w2d[i]
L2dex = (y_02dex + dt2d*L2dex)*phi2d*ufl.dx
linear_form2dex = fem.form(L2dex)

# Assemble FEM matrices.
C2d = fem.petsc.assemble_matrix(bilinear_form2d)
C2d.assemble()
d2d = fem.petsc.create_vector(linear_form2d)
d2dex = fem.petsc.create_vector(linear_form2dex)

# Create solver.
solver2d = PETSc.KSP().create(domain2d.comm)
solver2d.setOperators(C2d)
solver2d.setType(PETSc.KSP.Type.PREONLY)
solver2d.getPC().setType(PETSc.PC.Type.LU)

#Error function
e_W2d = fem.Function(V2d)
error_yn_ex_yn2d = fem.form(ufl.inner(e_W2d, e_W2d) * ufl.dx)
norm_yn_ex2d = fem.form(ufl.inner(y_n2dex, y_n2dex) * ufl.dx)
norm_yn2d = fem.form(ufl.inner(y_n2d, y_n2d) * ufl.dx)

#------------------------------------------------------------------------------------------------------------------------------
# Define PDE problem components for nl-----------------------------------------------------------------------------------------

# Define initial time.
tnl = 0

# Time intervall.
dtnl = (args.T_nl/args.num_steps_nl)

# Define space interval.

#1d
Omeganl = [-1,1]

# Define mesh.

# One dimensional mesh.
nxnl = 200
domainnl = mesh.create_interval(MPI.COMM_WORLD, nxnl, Omeganl)

# Define test function space.
Vnl = fem.functionspace(domainnl, ("P", 1))

# Define Dirichlet Boundary Condition. 
fdimnl = domainnl.topology.dim - 1
boundary_facetsnl = mesh.locate_entities_boundary(domainnl, fdimnl, lambda x: np.full(x.shape[1], True, dtype=bool))
bcnl = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(Vnl, fdimnl, boundary_facetsnl), Vnl)

class Initial_conditionnl():

    def __init__(self):
        pass

    def __call__(self,x):
        values = np.zeros(x.shape,dtype=PETSc.ScalarType)
        values = (x[0]-1)*(x[0]+1)+5
        return values

initial_conditionnl = Initial_conditionnl()

y_0nl = fem.Function(Vnl)
y_0nl.name = "y_0"
y_0nl.interpolate(initial_conditionnl)

# Trial and test function.
phinl = ufl.TestFunction(Vnl)

# PDE coefficients
nunl = 0.1

# Indicator function for 1d PDE also called acctuator functions.
w1nl = np.array([-0.7,-0.4])
w2nl = np.array([-0.2,0.2])
w3nl = np.array([0.4,0.7])
tol1d = 1e-14

acctuatorDomainsnl = [w1nl, w2nl, w3nl]

I_wnl = [fem.Function(Vnl) for i in range(0,len(acctuatorDomainsnl))]

for i, acct in enumerate(acctuatorDomainsnl):
  I_wnl[i].interpolate(lambda x: np.where(np.logical_and(x[0] - acct[0] +tol1d >= 0, acct[1] + tol1d - x[0] >= 0), 1, 0))
    
acctuatornl = np.array([I_wnl[0].x.array])
for i in range(1,len(I_wnl)):
  acctuatornl = np.r_[acctuatornl, [I_wnl[i].x.array]]

resortIndexnl = "empty"

# Define solution variable, and interpolate initial condition.
y_nnl = fem.Function(Vnl)
y_nnl.name = "y_n"
y_nnl.interpolate(initial_conditionnl)

# Control vector components.
usnl = [fem.Constant(domainnl, 0.0) for i in range(0,len(I_wnl))]

# Variational Problem (RHS).
Lnl = usnl[0]*I_wnl[0]
for i in range(1,len(usnl)):
  Lnl = Lnl + usnl[i]*I_wnl[i]
Lnl = (y_0nl + dtnl*Lnl)*phinl*ufl.dx
linear_formnl = fem.form(Lnl)

# Variational Problem (LHS).
a_hnl = y_nnl*phinl*ufl.dx + nunl*dtnl*ufl.dot(ufl.grad(y_nnl),ufl.grad(phinl))*ufl.dx - dtnl*y_nnl*(1-y_nnl**2)*phinl*ufl.dx - Lnl

# Create solver.
problemnl = femPetsc.NonlinearProblem(a_hnl, y_nnl)
solvernl = nlsPetsc.NewtonSolver(MPI.COMM_WORLD, problemnl)
solvernl.convergence_criterion = "incremental"
solvernl.rtol = 1e-6
solvernl.report = True

ksp = solvernl.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

#Error function
norm_ynnl = fem.form(ufl.inner(y_nnl, y_nnl) * ufl.dx)

#------------------------------------------------------------------------------------------------------------------------------
k = 0
for modelName in modelNames:

  savedModels = []

  if singleOutput == 1:
    savedModels = [args.savedModel]
  else:
    for savedModelFile in os.listdir(checkpoints_diretory_path(args.envId, modelName)):
      if savedModelFile!='fileDirectory.json':
        savedModels.append(savedModelFile)

  with open(checkpoints_fileDirectory(checkpoints_diretory_path(args.envId, modelName)),'r') as f:
      data = json.loads(f.read())

  for savedModel in savedModels:

    df_files = pd.json_normalize(data['fileNames'])
    modelFileInfo = df_files[df_files['fileName'] == savedModel]
    modelParFileId = modelFileInfo['parFileId'].tolist()[0]

    with open(path.join(parameter_directory_path(args.envId, modelName),"{}_{}.txt".format(modelName, modelParFileId))) as f:
      lines = f.readlines()
    parameter_list = [x[:len(x)-1].split(" = ") for x in lines]

    def getPar(name):
      par = [x[1] for x in parameter_list if x[0]==name]
      if len(par)==1:
        return par[0]
      else:
        return '0'

    # Read all nescesarry parameters.
    dimS = int(getPar('dimS'))
    dimA = int(getPar('dimA'))
    h = float(getPar('h'))
    ctrl_range = np.fromstring(getPar('ctrl_range')[1:-1], dtype=np.float64, sep=' ')
    env_id = getPar('env_id')
    useExactSolution = getPar('useExactSolution') == 'True'
    useMatCARE = getPar('useMatCARE') == 'True'
    Kpath = getPar('Kpath')
    model = getPar('model')
    Lc = float(getPar('L'))
    T = float(getPar('T'))
    time_steps = int(getPar('time_steps'))
    gamma = float(getPar('gamma')) 
    lr = float(getPar('lr'))
    actor_lr = float(getPar('actor_lr'))
    critic_lr = float(getPar('critic_lr'))
    sigma = float(getPar('sigma'))
    polyak = float(getPar('polyak'))
    hidden1 = int(getPar('hidden_size1'))
    hidden2 = int(getPar('hidden_size2'))
    buffer_size = int(float(getPar('buffer_size')))
    batch_size = int(getPar('batch_size'))
    smooth = getPar('smooth')=='True'
    double = getPar('double')=='True'
    h_scale = float(getPar('h_scale'))
    device = getPar('device')
    render = getPar('render')=='True'
    verboseLoopTraining = False
    # scale gamma & learning rate
    gamma = 1. - h_scale * (1. - gamma)
    lr = h_scale * lr

    if args.envId=='Linear1dPDEEnv-v0':
      T = args.T_l1
      time_steps = args.num_steps_l1      
    elif args.envId=='Linear2dPDEEnv-v0':
      T = args.T_l2
      time_steps = args.num_steps_l2
    elif args.envId=='NonLinearPDEEnv-v0':
      T = args.T_nl
      time_steps = args.num_steps_nl

    print('{},{},{}'.format(savedModel, device, k))
    k = k + 1

    if args.device==device:

      # Create environment of chosen model.
      env = scaled_env(env_id=env_id, T=T, time_steps=time_steps, useExactSolution=useExactSolution, useMatCARE=useMatCARE, Kpath=Kpath, scale_factor=h_scale)

      if args.envId=='Linear1dPDEEnv-v0':

        # Time step configuraton.
        num_steps = args.num_steps_l1    
        dt = args.T_l1 / num_steps

        if singleOutput==1:
          # Create pde output file-------------------------------------------------------------------------------------------------
          pde_solution = path.join(pdeSolutions_directory_path(args.envId, modelName),"{}_state.xdmf".format(savedModel))
          xdmf = io.XDMFFile(domain1d.comm, pde_solution, "w", encoding=io.XDMFFile.Encoding.HDF5)
          xdmf.write_mesh(domain1d)
          xdmf.write_function(y_n1d, t1d)

        # Create norm output file------------------------------------------------------------------------------------------------
        if not os.path.exists(stateAndControlNorms_directory_path(args.envId, modelName)+'/'):
          os.mkdir(stateAndControlNorms_directory_path(args.envId, modelName)+'/')
        stateAndControlNorms = path.join(stateAndControlNorms_directory_path(args.envId, modelName),"{}_norms.csv".format(savedModel))
        controlHistory = []

        # Create agent.
        if algo == 'ddpg':
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
                            hidden1=hidden1,
                            hidden2=hidden2,
                            acctuator=acctuator1d,
                            resortIndex=resortIndex1d,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            h_scale=h_scale,
                            device=device,
                            render=render)
        elif algo == 'hjdqn':
          agent = HJDQNAgent(dimS, dimA, ctrl_range,
                             gamma,
                             h, Lc, sigma,
                             verboseLoopTraining,
                             model,
                             acctuator1d,
                             resortIndex1d,
                             lr,
                             polyak,
                             buffer_size,
                             batch_size,
                             smooth=smooth,
                             device=device,
                             double=double,
                             render=render,
                             scale_factor=h_scale)

        # Load model and set resume training option.
        agent.load_model(path.join(checkpoints_diretory_path(args.envId, modelName), savedModel),resumeTraining=True)

        # Initial values for action network.
        noise = np.zeros(dimA)
        action = np.zeros(dimA)

        # Whole time intervall.
        for i in range(num_steps):
          t1d += dt

          state = y_01d.x.array
          state_exact = y_01dex.x.array

          # Control vector.
          if algo =='hjdqn':
            action = agent.get_action(state, action, noise)
          elif algo == 'ddpg':
            action = agent.get_action(state, eval=False)

          # Exact action
          action_exact = env.get_exact_action(state_exact)

          # Controls
          control = action.tolist()
          control_exact = action_exact.tolist()

          # Update coefficients of control vector.
          for i in range(0,len(us1d)):
            us1d[i].value = control[i]
          for i in range(0,len(us1dex)):
            us1dex[i].value = control_exact[i]

          # Update the right hand side reusing the initial vector
          with d1d.localForm() as loc_b:
            loc_b.set(0)
          fem.petsc.assemble_vector(d1d, linear_form1d)
          with d1dex.localForm() as loc_b:
            loc_b.set(0)
          fem.petsc.assemble_vector(d1dex, linear_form1dex)

          # Apply Dirichlet boundary condition to the vector
          fem.petsc.apply_lifting(d1d, [bilinear_form1d], [[bc1d]])
          d1d.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
          fem.petsc.set_bc(d1d, [bc1d])
          fem.petsc.apply_lifting(d1dex, [bilinear_form1d], [[bc1d]])
          d1dex.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
          fem.petsc.set_bc(d1dex, [bc1d])

          # Solve linear problem
          solver1d.solve(d1d, y_n1d.vector)
          y_n1d.x.scatter_forward()
          solver1d.solve(d1dex, y_n1dex.vector)
          y_n1dex.x.scatter_forward()

          # Update solution at previous time step
          y_01d.x.array[:] = y_n1d.x.array
          y_01dex.x.array[:] = y_n1dex.x.array

          if singleOutput==1:
            # Write solution to file
            xdmf.write_function(y_n1d, t1d)

          # Compute the l2 error between y_nex - y_n.
          e_W1d.x.array[:] = y_n1dex.x.array - y_n1d.x.array
          l2Error_ynex_yn = np.sqrt(domain1d.comm.allreduce(fem.assemble_scalar(error_yn_ex_yn1d), op=MPI.SUM))

          # Compute the l2 norms of y_nex and y_n
          l2Norm_yn_ex = np.sqrt(domain1d.comm.allreduce(fem.assemble_scalar(norm_yn_ex1d), op=MPI.SUM))
          l2Norm_yn = np.sqrt(domain1d.comm.allreduce(fem.assemble_scalar(norm_yn1d), op=MPI.SUM))

          # Save norms
          controlHistory.append([t1d] + [l2Error_ynex_yn, l2Norm_yn_ex, l2Norm_yn] + [abs(ctrl[0] - ctrl[1]) for ctrl in zip(control_exact, control)] + [abs(ctrl) for ctrl in control]+[abs(ctrl_exact) for ctrl_exact in control_exact])
        if singleOutput==1:
          xdmf.close()

        controlHistoryArray = np.array(controlHistory)
        dfNorms = pd.DataFrame({'t': controlHistoryArray[:, 0], 'L2ErrorStateExState': controlHistoryArray[:, 1], 'L2NormStateEx': controlHistoryArray[:, 2], 'L2NormState': controlHistoryArray[:, 3], 
                                'Absu1exu1': controlHistoryArray[:, 4], 'Absu2exu2': controlHistoryArray[:, 5], 'Absu3exu3': controlHistoryArray[:, 6], 
                                'Absu1': controlHistoryArray[:, 7], 'Absu2': controlHistoryArray[:, 8], 'Absu3': controlHistoryArray[:, 9], 
                                'Absu1ex': controlHistoryArray[:, 10], 'Absu2ex': controlHistoryArray[:, 11], 'Absu3ex': controlHistoryArray[:, 12]})
        dfNorms.to_csv(stateAndControlNorms)
        y_01d.interpolate(initial_condition1d)
        y_n1d.interpolate(initial_condition1d)
        y_01dex.interpolate(initial_condition1d)
        y_n1dex.interpolate(initial_condition1d)
        env.reset()
        t1d = 0

      if args.envId=='Linear2dPDEEnv-v0':

        # Time step configuraton.
        num_steps = args.num_steps_l2   
        dt = args.T_l2 / num_steps

        if singleOutput==1:
          # Create pde output file-------------------------------------------------------------------------------------------------
          pde_solution = path.join(pdeSolutions_directory_path(args.envId, modelName),"{}_state.xdmf".format(savedModel))
          xdmf = io.XDMFFile(domain2d.comm, pde_solution, "w", encoding=io.XDMFFile.Encoding.HDF5)
          xdmf.write_mesh(domain2d)
          xdmf.write_function(y_n2d, t2d)

        # Create norm output file------------------------------------------------------------------------------------------------
        if not os.path.exists(stateAndControlNorms_directory_path(args.envId, modelName)+'/'):
          os.mkdir(stateAndControlNorms_directory_path(args.envId, modelName)+'/')
        stateAndControlNorms = path.join(stateAndControlNorms_directory_path(args.envId, modelName),"{}_norms.csv".format(savedModel))
        controlHistory = []

        # Create agent.
        if algo == 'ddpg':
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
                            hidden1=hidden1,
                            hidden2=hidden2,
                            acctuator=acctuator2d,
                            resortIndex=resortIndex2d,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            h_scale=h_scale,
                            device=device,
                            render=render)
        elif algo == 'hjdqn':
          agent = HJDQNAgent(dimS, dimA, ctrl_range,
                             gamma,
                             h, Lc, sigma,
                             verboseLoopTraining,
                             model,
                             acctuator2d,
                             resortIndex2d,
                             lr,
                             polyak,
                             buffer_size,
                             batch_size,
                             smooth=smooth,
                             device=device,
                             double=double,
                             render=render,
                             scale_factor=h_scale)

        # Load model and set resume training option.
        agent.load_model(path.join(checkpoints_diretory_path(args.envId, modelName), savedModel),resumeTraining=True)

        # Initial values for action network.
        noise = np.zeros(dimA)
        action = np.zeros(dimA)

        # Whole time intervall.
        for i in range(num_steps):
          t2d += dt

          state = y_02d.x.array
          state_exact = y_02dex.x.array

          # Control vector.
          if algo =='hjdqn':
            action = agent.get_action(state, action, noise)
          elif algo == 'ddpg':
            action = agent.get_action(state, eval=False)

          # Exact action
          action_exact = env.get_exact_action(state_exact)

          # Controls
          control = action.tolist()
          control_exact = action_exact.tolist()

          # Update coefficients of control vector.
          for i in range(0,len(us2d)):
            us2d[i].value = control[i]
          for i in range(0,len(us2dex)):
            us2dex[i].value = control_exact[i]

          # Update the right hand side reusing the initial vector
          with d2d.localForm() as loc_b:
            loc_b.set(0)
          fem.petsc.assemble_vector(d2d, linear_form2d)
          with d2dex.localForm() as loc_b:
            loc_b.set(0)
          fem.petsc.assemble_vector(d2dex, linear_form2dex)

          # Apply Dirichlet boundary condition to the vector
          fem.petsc.apply_lifting(d2d, [bilinear_form2d], [[bc2d]])
          d2d.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
          fem.petsc.set_bc(d2d, [bc2d])
          fem.petsc.apply_lifting(d2dex, [bilinear_form2d], [[bc2d]])
          d2dex.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
          fem.petsc.set_bc(d2dex, [bc2d])

          # Solve linear problem
          solver2d.solve(d2d, y_n2d.vector)
          y_n2d.x.scatter_forward()
          solver2d.solve(d2dex, y_n2dex.vector)
          y_n2dex.x.scatter_forward()

          # Update solution at previous time step
          y_02d.x.array[:] = y_n2d.x.array
          y_02dex.x.array[:] = y_n2dex.x.array

          if singleOutput==1:
            # Write solution to file
            xdmf.write_function(y_n2d, t2d)

          # Compute the l2 error between y_nex - y_n.
          e_W2d.x.array[:] = y_n2dex.x.array - y_n2d.x.array
          l2Error_ynex_yn = np.sqrt(domain2d.comm.allreduce(fem.assemble_scalar(error_yn_ex_yn2d), op=MPI.SUM))

          # Compute the l2 norms of y_nex and y_n
          l2Norm_yn_ex = np.sqrt(domain2d.comm.allreduce(fem.assemble_scalar(norm_yn_ex2d), op=MPI.SUM))
          l2Norm_yn = np.sqrt(domain2d.comm.allreduce(fem.assemble_scalar(norm_yn2d), op=MPI.SUM))

          # Save norms
          controlHistory.append([t2d] + [l2Error_ynex_yn, l2Norm_yn_ex, l2Norm_yn] + [abs(ctrl[0] - ctrl[1]) for ctrl in zip(control_exact, control)] + [abs(ctrl) for ctrl in control]+[abs(ctrl_exact) for ctrl_exact in control_exact])
        if singleOutput==1:
          xdmf.close()

        controlHistoryArray = np.array(controlHistory)
        dfNorms = pd.DataFrame({'t': controlHistoryArray[:, 0], 'L2ErrorStateExState': controlHistoryArray[:, 1], 'L2NormStateEx': controlHistoryArray[:, 2], 'L2NormState': controlHistoryArray[:, 3], 
                               'Absu1exu1': controlHistoryArray[:, 4], 'Absu2exu2': controlHistoryArray[:, 5], 'Absu3exu3': controlHistoryArray[:, 6],
                               'Absu4exu4': controlHistoryArray[:, 7], 'Absu5exu5': controlHistoryArray[:, 8], 'Absu6exu6': controlHistoryArray[:, 9],
                               'Absu7exu7': controlHistoryArray[:, 10], 'Absu8exu8': controlHistoryArray[:, 11], 'Absu9exu9': controlHistoryArray[:, 12],
                               'Absu1': controlHistoryArray[:, 13], 'Absu2': controlHistoryArray[:, 14], 'Absu3': controlHistoryArray[:, 15],
                               'Absu4': controlHistoryArray[:, 16], 'Absu5': controlHistoryArray[:, 17], 'Absu6': controlHistoryArray[:, 18],
                               'Absu7': controlHistoryArray[:, 19], 'Absu8': controlHistoryArray[:, 20], 'Absu9': controlHistoryArray[:, 21],                               
                               'Absu1ex': controlHistoryArray[:, 22], 'Absu2ex': controlHistoryArray[:, 23], 'Absu3ex': controlHistoryArray[:, 24],
                               'Absu4ex': controlHistoryArray[:, 25], 'Absu5ex': controlHistoryArray[:, 26], 'Absu6ex': controlHistoryArray[:, 27],
                               'Absu7ex': controlHistoryArray[:, 28], 'Absu8ex': controlHistoryArray[:, 29], 'Absu9ex': controlHistoryArray[:, 30]})
        dfNorms.to_csv(stateAndControlNorms)
        y_02d.interpolate(initial_condition2d)
        y_n2d.interpolate(initial_condition2d)
        y_02dex.interpolate(initial_condition2d)
        y_n2dex.interpolate(initial_condition2d)
        env.reset()
        t2d = 0

      if args.envId=='NonLinearPDEEnv-v0':

        # Time step configuraton.
        num_steps = args.num_steps_nl    
        dt = args.T_nl / num_steps

        if singleOutput==1:
          # Create pde output file-------------------------------------------------------------------------------------------------
          pde_solution = path.join(pdeSolutions_directory_path(args.envId, modelName),"{}_state.xdmf".format(savedModel))
          xdmf = io.XDMFFile(domainnl.comm, pde_solution, "w", encoding=io.XDMFFile.Encoding.HDF5)
          xdmf.write_mesh(domainnl)
          xdmf.write_function(y_nnl, tnl)

        # Create norm output file------------------------------------------------------------------------------------------------
        if not os.path.exists(stateAndControlNorms_directory_path(args.envId, modelName)+'/'):
          os.mkdir(stateAndControlNorms_directory_path(args.envId, modelName)+'/')
        stateAndControlNorms = path.join(stateAndControlNorms_directory_path(args.envId, modelName),"{}_norms.csv".format(savedModel))
        controlHistory = []

        # Create agent.
        if algo == 'ddpg':
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
                            hidden1=hidden1,
                            hidden2=hidden2,
                            acctuator=acctuatornl,
                            resortIndex=resortIndexnl,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            h_scale=h_scale,
                            device=device,
                            render=render)
        elif algo == 'hjdqn':
          agent = HJDQNAgent(dimS, dimA, ctrl_range,
                             gamma,
                             h, Lc, sigma,
                             verboseLoopTraining,
                             model,
                             acctuatornl,
                             resortIndexnl,
                             lr,
                             polyak,
                             buffer_size,
                             batch_size,
                             smooth=smooth,
                             device=device,
                             double=double,
                             render=render,
                             scale_factor=h_scale)

        # Load model and set resume training option.
        agent.load_model(path.join(checkpoints_diretory_path(args.envId, modelName), savedModel),resumeTraining=True)

        # Initial values for action network.
        noise = np.zeros(dimA)
        action = np.zeros(dimA)

        # Whole time intervall.
        for i in range(num_steps):
          tnl += dt

          state = y_0nl.x.array

          # Control vector.
          if algo =='hjdqn':
            action = agent.get_action(state, action, noise)
          elif algo == 'ddpg':
            action = agent.get_action(state, eval=False)

          # Controls
          control = action.tolist()

          # Update coefficients of control vector.
          for i in range(0,len(usnl)):
            usnl[i].value = control[i]

          # Solve linear problem
          _, converged = solvernl.solve(y_nnl)
          y_nnl.x.scatter_forward()
          assert(converged)

          # Update solution at previous time step
          y_0nl.x.array[:] = y_nnl.x.array

          if singleOutput==1:
            # Write solution to file
            xdmf.write_function(y_nnl, tnl)

          # Compute the l2 norms of y_nex and y_n
          l2Norm_yn = np.sqrt(domainnl.comm.allreduce(fem.assemble_scalar(norm_ynnl), op=MPI.SUM))

          # Save norms
          controlHistory.append([tnl] + [l2Norm_yn] + [abs(ctrl) for ctrl in control])
        if singleOutput==1:
          xdmf.close()

        controlHistoryArray = np.array(controlHistory)
        dfNorms = pd.DataFrame({'t': controlHistoryArray[:, 0], 'L2NormState': controlHistoryArray[:, 1], 'Absu1': controlHistoryArray[:, 2], 'Absu2': controlHistoryArray[:, 3], 'Absu3': controlHistoryArray[:, 4]})
        dfNorms.to_csv(stateAndControlNorms)
        y_0nl.interpolate(initial_conditionnl)
        y_nnl.interpolate(initial_conditionnl)
        env.reset()
        tnl = 0