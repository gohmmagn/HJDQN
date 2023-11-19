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
parser.add_argument('--modelName', required=True)
parser.add_argument('--savedModel', required=True)

args = parser.parse_args()
algo = args.algo
pdeSolutions_directory_path = "outputs/{}/{}/pde_solutions".format(args.envId, args.modelName)
parameter_directory_path = "outputs/{}/{}/parameter".format(args.envId, args.modelName)
checkpoints_diretory_path = "outputs/{}/{}/checkpoints".format(args.envId, args.modelName)
checkpoints_fileDirectory = path.join(checkpoints_diretory_path,'fileDirectory.json')

with open(checkpoints_fileDirectory,'r') as f:
            data = json.loads(f.read())
df_files = pd.json_normalize(data['fileNames'])
modelFileInfo = df_files[df_files['fileName'] == args.savedModel]
modelParFileId = modelFileInfo['parFileId'].tolist()[0]

with open(path.join(parameter_directory_path,"{}_{}.txt".format(args.modelName,modelParFileId))) as f:
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

# Create environment of chosen model.
env = scaled_env(env_id=env_id, T=T, time_steps=time_steps, useExactSolution=useExactSolution, useMatCARE=useMatCARE, Kpath=Kpath, scale_factor=h_scale)
acctuator = env.unwrapped.acctuator
resortIndex = env.unwrapped.resortIndex

if args.envId=='Linear1dPDEEnv-v0':

  # Define PDE problem components----------------------------------------------------------------------------------------------------------------------------------------------

  # Define initial time.
  t = 0

  # Define space interval.

  #1d
  Omega = [-1,1]

  # Time step configuraton.
  num_steps = time_steps    
  dt = T / num_steps

  # Define mesh.

  # One dimensional mesh.
  nx = 200
  domain = mesh.create_interval(MPI.COMM_WORLD, nx, Omega)

  # Define test function space.
  V = fem.functionSpace(domain, ("P", 1))

  # Define Dirichlet Boundary Condition. 
  fdim = domain.topology.dim - 1
  boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
  bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

  # Create initial condition for the state function and interpolate it in V.

  #class Initial_condition():

  #  def __init__(self):
  #      pass

  #  def __call__(self,x):
  #      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
  #      values = (x[0]-1)*(x[0]+1)+5
  #      return values

  #initial_condition = Initial_condition()

  def initial_condition(x):
    return (x[0]-1)*(x[0]+1)+5

  y_0 = fem.Function(V)
  y_0.name = "y_0"
  y_0.interpolate(initial_condition)

  # Trial and test function.
  y, phi = ufl.TrialFunction(V), ufl.TestFunction(V)

  # PDE coefficients
  nu = 0.01

  #class PDE_a():

  #  def __init__(self):
  #      pass

  #  def __call__(self,x):
  #      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
  #      values = np.ones(x[0].shape)
  #      return values

  #class PDE_b():

  #  def __init__(self):
  #      pass

  #  def __call__(self,x):
  #      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
  #      values = np.ones(x[0].shape)
  #      return values

  #a_V = fem.Function(V)
  #a_V.interpolate(PDE_a())

  #b_V1 = fem.Function(V)
  #b_V1.interpolate(PDE_b())
  #b_V = ufl.as_vector([b_V1])

  a_V = 1.0
  b_V = ufl.as_vector([1.0])

  # Indicator function for 1d PDE also called acctuator functions.
  w1 = np.array([-0.7,-0.4])
  w2 = np.array([-0.2,0.2])
  w3 = np.array([0.4,0.7])
  tol = 1e-14

  acctuatorDomains = [w1, w2, w3]

  I_w = [fem.Function(V) for i in range(0,len(acctuatorDomains))]

  for i, acct in enumerate(acctuatorDomains):
    I_w[i].interpolate(lambda x: np.where(np.logical_and(x[0] - acct[0] +tol >= 0, acct[1] + tol - x[0] >= 0), 1, 0))
    
  acctuator = np.array([I_w[0].x.array])
  for i in range(1,len(I_w)):
    acctuator = np.r_[acctuator, [I_w[i].x.array]]

  resortIndex = "empty"

  # Create output file---------------------------------------------------------------------------------------------------------------------------------------------------------
  pde_solution = path.join(pdeSolutions_directory_path,"{}_state.xdmf".format(args.savedModel))
  xdmf = io.XDMFFile(domain.comm, pde_solution, "w", encoding=io.XDMFFile.Encoding.HDF5)
  xdmf.write_mesh(domain)

  # Define solution variable, and interpolate initial condition.
  y_n = fem.Function(V)
  y_n.name = "y_n"
  y_n.interpolate(initial_condition)
  xdmf.write_function(y_n, t)

  # Control vector components.
  us = [fem.Constant(domain, 0.0) for i in range(0,len(I_w))]

  # Variational Problem (LHS).
  a_h = y*phi*ufl.dx + nu*dt*ufl.dot(ufl.grad(y),ufl.grad(phi))*ufl.dx + dt*a_V*y*phi*ufl.dx - dt*ufl.dot(b_V*y,ufl.grad(phi))*ufl.dx
  bilinear_form = fem.form(a_h)

  # Variational Problem (RHS).
  L = us[0]*I_w[0]
  for i in range(1,len(us)):
    L = L + us[i]*I_w[i]
  L = (y_0 + dt*L)*phi*ufl.dx
  linear_form = fem.form(L)

  # Assemble FEM matrices.
  C = fem.petsc.assemble_matrix(bilinear_form)
  C.assemble()
  d = fem.petsc.create_vector(linear_form)

  # Create solver.
  solver = PETSc.KSP().create(domain.comm)
  solver.setOperators(C)
  solver.setType(PETSc.KSP.Type.PREONLY)
  solver.getPC().setType(PETSc.PC.Type.LU)
  
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
                       acctuator=acctuator,
                       resortIndex=resortIndex,
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

  # Load model and set resume training option.
  agent.load_model(path.join(checkpoints_diretory_path, args.savedModel),resumeTraining=True)

  # Initial values for action network.
  state = y_0.x.array
  noise = np.zeros(dimA)
  action = np.zeros(dimA)

  # Whole time intervall.
  for i in range(num_steps):
    t += dt

    # Control vector.
    if algo =='hjdqn':
       action = agent.get_action(state, action, noise)
    elif algo == 'ddpg':
       action = agent.get_action(state, eval=False)
    control = action.tolist()

    # Update coefficients of control vector.
    for i in range(0,len(us)):
      us[i].value = control[i]

    # Update the right hand side reusing the initial vector
    with d.localForm() as loc_b:
         loc_b.set(0)
    fem.petsc.assemble_vector(d, linear_form)

    # Apply Dirichlet boundary condition to the vector
    fem.petsc.apply_lifting(d, [bilinear_form], [[bc]])
    d.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(d, [bc])

    # Solve linear problem
    solver.solve(d, y_n.vector)
    y_n.x.scatter_forward()

    # Update solution at previous time step
    y_0.x.array[:] = y_n.x.array

    # Write solution to file
    xdmf.write_function(y_n, t)
  xdmf.close()

if args.envId=='Linear2dPDEEnv-v0':

  # Define PDE problem components----------------------------------------------------------------------------------------------------------------------------------------------

  # Define initial time.
  t = 0

  # Define space interval.

  #1d
  Omega = [np.array([0,0]), np.array([1,1])]

  # Time step configuraton.
  num_steps = time_steps    
  dt = T / num_steps

  # Define mesh.

  # One dimensional mesh.
  nx = 32
  ny = 32
  domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

  # Define test function space.
  V = fem.functionSpace(domain, ("P", 1))

  # Define Dirichlet Boundary Condition. 
  fdim = domain.topology.dim - 1
  boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
  bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
  
    # Resorting index to reshape 1d output vectors.
  mesh_coords = np.transpose(domain.geometry.x[:,0:2])
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
  
  resortIndex = IY

  # Create initial condition for the state function and interpolate it in V.

  class Initial_condition():

    def __init__(self):
      pass

    def __call__(self,x):
       values = np.zeros(x.shape,dtype=PETSc.ScalarType)
       values = 3*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
       return values

  initial_condition = Initial_condition()

  y_0 = fem.Function(V)
  y_0.name = "y_0"
  y_0.interpolate(initial_condition)

  # Trial and test function.
  y, phi = ufl.TrialFunction(V), ufl.TestFunction(V)

  # PDE coefficients
  nu = 0.01

  class PDE_a():

    def __init__(self):
      pass

    def __call__(self,x):
      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
      values = x[0]+x[1]
      return values

  class PDE_b_1():

    def __init__(self):
      pass

    def __call__(self,x):
      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
      values = x[0]
      return values

  class PDE_b_2():

    def __init__(self):
      pass

    def __call__(self,x):
      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
      values = x[1]
      return values

  a_V = fem.Function(V)
  a_V.interpolate(PDE_a())

  b_V1 = fem.Function(V)
  b_V2 = fem.Function(V)
  b_V1.interpolate(PDE_b_1())
  b_V2.interpolate(PDE_b_2())
  b_V = ufl.as_vector([b_V1, b_V2])

  # Indicator function for 1d PDE also called acctuator functions.
  w1 = np.array([[0.1,0.3],[0.1,0.3]])
  w2 = np.array([[0.4,0.6],[0.1,0.3]])
  w3 = np.array([[0.7,0.9],[0.1,0.3]])
  w4 = np.array([[0.1,0.3],[0.4,0.6]])
  w5 = np.array([[0.4,0.6],[0.4,0.6]])
  w6 = np.array([[0.7,0.9],[0.4,0.6]])
  w7 = np.array([[0.1,0.3],[0.7,0.9]])
  w8 = np.array([[0.4,0.6],[0.7,0.9]])
  w9 = np.array([[0.7,0.9],[0.7,0.9]])
  tol = 1e-14

  acctuatorDomains = [w1, w2, w3, w4, w5, w6, w7, w8, w9]

  I_w = [fem.Function(V) for i in range(0,len(acctuatorDomains))]

  for i, acct in enumerate(acctuatorDomains):
    I_w[i].interpolate(lambda x: np.where(np.logical_and(np.logical_and(x[0]>=acct[0,0], x[0]<=acct[0,1]),np.logical_and(x[1]>=acct[1,0], x[1]<=acct[1,1])), 1, 0))

  acctuator = np.array([I_w[0].x.array])
  for i in range(1,len(I_w)):
    acctuator = np.r_[acctuator, [I_w[i].x.array]]

  # Create output file---------------------------------------------------------------------------------------------------------------------------------------------------------
  pde_solution = path.join(pdeSolutions_directory_path,"{}_state.xdmf".format(args.savedModel))
  xdmf = io.XDMFFile(domain.comm, pde_solution, "w", encoding=io.XDMFFile.Encoding.HDF5)
  xdmf.write_mesh(domain)

  # Define solution variable, and interpolate initial condition.
  y_n = fem.Function(V)
  y_n.name = "y_n"
  y_n.interpolate(initial_condition)
  xdmf.write_function(y_n, t)

  # Control vector components.
  us = [fem.Constant(domain, 0.0) for i in range(0,len(I_w))]

  # Variational Problem (LHS).
  a_h = y*phi*ufl.dx + nu*dt*ufl.dot(ufl.grad(y),ufl.grad(phi))*ufl.dx + dt*a_V*y*phi*ufl.dx - dt*ufl.dot(b_V*y,ufl.grad(phi))*ufl.dx
  bilinear_form = fem.form(a_h)

  # Variational Problem (RHS).
  L = us[0]*I_w[0]
  for i in range(1,len(us)):
    L = L + us[i]*I_w[i]
  L = (y_0 + dt*L)*phi*ufl.dx
  linear_form = fem.form(L)

  # Assemble FEM matrices.
  C = fem.petsc.assemble_matrix(bilinear_form)
  C.assemble()
  d = fem.petsc.create_vector(linear_form)

  # Create solver.
  solver = PETSc.KSP().create(domain.comm)
  solver.setOperators(C)
  solver.setType(PETSc.KSP.Type.PREONLY)
  solver.getPC().setType(PETSc.PC.Type.LU)
  
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
                       acctuator=acctuator,
                       resortIndex=resortIndex,
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

  # Load model and set resume training option.
  agent.load_model(path.join(checkpoints_diretory_path, args.savedModel),resumeTraining=True)

  # Initial values for action network.
  state = y_0.x.array
  noise = np.zeros(dimA)
  action = np.zeros(dimA)

  # Whole time intervall.
  for i in range(num_steps):
    t += dt

    # Control vector.
    if algo =='hjdqn':
       action = agent.get_action(state, action, noise)
    elif algo == 'ddpg':
       action = agent.get_action(state, eval=False)
    control = action.tolist()

    # Update coefficients of control vector.
    for i in range(0,len(us)):
      us[i].value = control[i]

    # Update the right hand side reusing the initial vector
    with d.localForm() as loc_b:
         loc_b.set(0)
    fem.petsc.assemble_vector(d, linear_form)

    # Apply Dirichlet boundary condition to the vector
    fem.petsc.apply_lifting(d, [bilinear_form], [[bc]])
    d.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(d, [bc])

    # Solve linear problem
    solver.solve(d, y_n.vector)
    y_n.x.scatter_forward()

    # Update solution at previous time step
    y_0.x.array[:] = y_n.x.array

    # Write solution to file
    xdmf.write_function(y_n, t)
  xdmf.close()

if args.envId=='NonLinearPDEEnv-v0':

  # One dimensional PDE.

  # Define initial time.
  t = 0

  # Define space interval.

  #1d
  Omega = [-1,1]

  # Time step configuraton.
  num_steps = time_steps
  dt = T / num_steps

  # Define mesh.

  # One dimensional mesh.
  nx = 200
  domain = mesh.create_interval(MPI.COMM_WORLD, nx, Omega)

  # Define test function space.
  V = fem.functionSpace(domain, ("P", 1))

  # Create initial condition for the state function and interpolate it in V.

  #class Initial_condition():
  #
  #  def __init__(self):
  #      pass
  #
  #  def __call__(self,x):
  #      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
  #      values = (x[0]-1)*(x[0]+1)+5
  #      return values

  #initial_condition = Initial_condition()

  def initial_condition(x):
    return (x[0]-1)*(x[0]+1)+5

  y_0 = fem.Function(V)
  y_0.name = "y_0"
  y_0.interpolate(initial_condition)

  # Test function.
  phi = ufl.TestFunction(V)

  # PDE coefficients
  nu = 0.1

  # Indicator function for 1d PDE also called acctuator functions.
  w1 = np.array([-0.7,-0.4])
  w2 = np.array([-0.2,0.2])
  w3 = np.array([0.4,0.7])
  tol = 1e-14

  acctuatorDomains = [w1, w2, w3]

  I_w = [fem.Function(V) for i in range(0,len(acctuatorDomains))]

  for i, acct in enumerate(acctuatorDomains):
    I_w[i].interpolate(lambda x: np.where(np.logical_and(x[0] - acct[0] +tol >= 0, acct[1] + tol - x[0] >= 0), 1, 0))

  # Create output file.
  pde_solution = path.join(pdeSolutions_directory_path,"{}_state.xdmf".format(args.savedModel))
  xdmf = io.XDMFFile(domain.comm, pde_solution, "w", encoding=io.XDMFFile.Encoding.HDF5)
  xdmf.write_mesh(domain)

  # Define solution variable, and interpolate initial solution for visualization in Paraview.
  y_n = fem.Function(V)
  y_n.name = "y_n"
  y_n.interpolate(initial_condition)
  xdmf.write_function(y_n, t)

  # Control vector components.
  us = [fem.Constant(domain, 0.0) for i in range(0,len(I_w))]

  # Variational Problem (RHS).
  L = us[0]*I_w[0]
  for i in range(1,len(us)):
    L = L + us[i]*I_w[i]
  L = (y_0 + dt*L)*phi*ufl.dx

  # Variational Problem.
  a_h = y_n*phi*ufl.dx + nu*dt*ufl.dot(ufl.grad(y_n),ufl.grad(phi))*ufl.dx - dt*y_n*(1-y_n**2)*phi*ufl.dx - L

  # Create solver.
  problem = femPetsc.NonlinearProblem(a_h, y_n)
  solver = nlsPetsc.NewtonSolver(MPI.COMM_WORLD, problem)
  solver.convergence_criterion = "incremental"
  solver.rtol = 1e-6
  solver.report = True

  ksp = solver.krylov_solver
  opts = PETSc.Options()
  option_prefix = ksp.getOptionsPrefix()
  opts[f"{option_prefix}ksp_type"] = "cg"
  opts[f"{option_prefix}pc_type"] = "gamg"
  opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
  ksp.setFromOptions()

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
                       acctuator=acctuator,
                       resortIndex=resortIndex,
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

  # Load model and set resume training option.
  agent.load_model(path.join(checkpoints_diretory_path, args.savedModel),resumeTraining=True)

  # Initial values for action network.
  state = y_0.x.array
  noise = np.zeros(dimA)
  action = np.zeros(dimA)

  # Whole time intervall.
  for i in range(num_steps):
    t += dt

    # Control vector.
    if algo =='hjdqn':
       action = agent.get_action(state, action, noise)
    elif algo == 'ddpg':
       action = agent.get_action(state, eval=False)
    control = action.tolist()

    # Update coefficients of control vector.
    for i in range(0,len(us)):
      us[i].value = control[i]

    # Solve linear problem
    _, converged = solver.solve(y_n)
    y_n.x.scatter_forward()
    assert(converged)

    # Update solution at previous time step
    y_0.x.array[:] = y_n.x.array

    # Write solution to file
    xdmf.write_function(y_n, t)
  xdmf.close()    