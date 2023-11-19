import control
import importlib
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import dolfinx
import ufl
from mpi4py import MPI
import petsc4py
petsc4py.init()
from petsc4py import PETSc
from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem import petsc as femPetsc
from dolfinx.nls import petsc as nlsPetsc

class NonLinearPDEEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, T=2.0, time_steps=200, useExactSolution=False ,useMatCARE=False, Kpath='empty'):
        super(NonLinearPDEEnv, self).__init__()

        # Set default env variables -----------------------------------------------------------------------------------------

        self.useExactSolution = useExactSolution
        self.useMatCARE = useMatCARE
        self.acctuator = 'empty'
        self.resortIndex = 'empty'

        #--------------------------------------------------------------------------------------------------------------------

        # Time discretization.-----------------------------------------------------------------------------------------------

        # Define initial and end time value.
        self.T = T

        # Time step configuraton.
        num_steps = time_steps
        self.dt = self.T / num_steps

        #-------------------------------------------------------------------------------------------------------------------

        # Space dicretization.----------------------------------------------------------------------------------------------

        # One dimensional domain.
        self.Omega = [-1,1]

        #One dimensional mesh.
        self.nx = 200
        self.domain = mesh.create_interval(MPI.COMM_WORLD, self.nx, self.Omega)

        # Define test function space.
        self.V = fem.functionSpace(self.domain, ("P", 1))

        #-----------------------------------------------------------------------------------------------------------------

        # Iteration variables --------------------------------------------------------------------------------------------

        # Create initial condition for the state function and interpolate it in V.

        #class Initial_condition():

        #  def __init__(self):
        #      pass

        #  def __call__(self,x):
        #      values = np.zeros(x.shape,dtype=PETSc.ScalarType)
        #      values = (x[0]-1)*(x[0]+1)+5
        #      return values

        #initial_condition = Initial_condition()

        self.y_0 = fem.Function(self.V)
        self.y_0.name = "y_0"
        self.y_0.interpolate(self.initial_condition)

        self.y_n = fem.Function(self.V)
        self.y_n.name = "y_n"
        self.y_n.interpolate(self.initial_condition)

        #-----------------------------------------------------------------------------------------------------------------

        # Variational Problem (Acctuator functions and coefficients) -----------------------------------------------------

        # PDE coefficients
        self.nu = 0.1

        # Indicator function for 1d PDE also called acctuator functions.
        w1 = np.array([-0.7,-0.4])
        w2 = np.array([-0.2,0.2])
        w3 = np.array([0.4,0.7])
        tol = 1e-14

        acctuatorDomains = [w1, w2, w3]

        self.I_w = [fem.Function(self.V) for i in range(0,len(acctuatorDomains))]

        for i, acct in enumerate(acctuatorDomains):
          self.I_w[i].interpolate(lambda x: np.where(np.logical_and(x[0] - acct[0] +tol >= 0, acct[1] + tol - x[0] >= 0), 1, 0))

        acctuator = np.array([self.I_w[0].x.array])
        for i in range(1,len(self.I_w)):
          acctuator = np.r_[acctuator, [self.I_w[i].x.array]]
        self.acctuator = acctuator

        #-----------------------------------------------------------------------------------------------------------------

        # Trial and test functions.---------------------------------------------------------------------------------------
        self.y_trial, self.phi = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        #-----------------------------------------------------------------------------------------------------------------

        # Linear form L---------------------------------------------------------------------------------------------------

        # Control vector components.
        self.us = [fem.Constant(self.domain, 0.0) for i in range(0,len(self.I_w))]

        self.L = self.us[0]*self.I_w[0]
        for i in range(1,len(self.us)):
          self.L = self.L + self.us[i]*self.I_w[i]
        self.L = (self.y_0 + self.dt*self.L)*self.phi*ufl.dx

        #-----------------------------------------------------------------------------------------------------------------

        # Variational form a_h--------------------------------------------------------------------------------------------

        self.a_h = self.y_n*self.phi*ufl.dx + self.nu*self.dt*ufl.dot(ufl.grad(self.y_n),ufl.grad(self.phi))*ufl.dx - self.dt*self.y_n*(1-self.y_n**2)*self.phi*ufl.dx - self.L

        #-----------------------------------------------------------------------------------------------------------------

        # Create solver---------------------------------------------------------------------------------------------------

        self.problem = femPetsc.NonlinearProblem(self.a_h, self.y_n)
        self.solver = nlsPetsc.NewtonSolver(MPI.COMM_WORLD, self.problem)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-6
        self.solver.report = False
        self.solver.error_on_nonconvergence = False

        self.ksp = self.solver.krylov_solver
        self.opts = PETSc.Options()
        self.option_prefix = self.ksp.getOptionsPrefix()
        self.opts[f"{self.option_prefix}ksp_type"] = "cg"
        self.opts[f"{self.option_prefix}pc_type"] = "gamg"
        self.opts[f"{self.option_prefix}pc_factor_mat_solver_type"] = "mumps"
        self.ksp.setFromOptions()
        
        #-----------------------------------------------------------------------------------------------------------------

        # Cost functional matrices ---------------------------------------------------------------------------------------

        # Mass matrix.
        mass_form = self.y_trial*self.phi*ufl.dx
        M_dx = femPetsc.assemble_matrix(fem.form(mass_form))
        M_dx.assemble()
        M_dx.convert("dense")
        self.M = M_dx.getDenseArray()

        # State dimension.
        self.n = self.M.shape[1]

        # Action dimesion.
        self.m = len(acctuatorDomains)

        # Matrices and coefficients for the reward function r = alpha*y'*Q*y + beta*u'*R*u.
        self.alpha = 1.
        self.beta = 0.01
        self.Q = self.alpha*self.M
        self.R = self.beta*np.eye(self.m)

        #-----------------------------------------------------------------------------------------------------------------

        # Initial control.------------------------------------------------------------------------------------------------

        self.initial_action = np.zeros((len(acctuatorDomains),))

        # Action and state spaces ----------------------------------------------------------------------------------------

        # Random sampling from the action space : $U[a, b)$.
        
        self.action_space = spaces.Box(
            low=-1000.0,
            high=1000.0, shape=(self.m,),
            dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0, shape=(self.n,),
            dtype=np.float64
        )

        #-----------------------------------------------------------------------------------------------------------------

        # State solution variables ---------------------------------------------------------------------------------------

        self.x = self.y_0.x.array

    def step(self, u):
    
        # Quadratic cost function alpha*x'*Q*x + beta*u'*R*u.
        costs = np.sum(np.dot(np.matmul(np.transpose(self.x),self.Q),self.x)) + np.sum(np.dot(np.matmul(np.transpose(u),self.R),u))

        # Update coefficients of control vector.
        uls = u.tolist()
        for i in range(0,len(self.us)):
          self.us[i].value = uls[i]

        # Solve linear problem.
        self.solver.solve(self.y_n)
        self.y_n.x.scatter_forward()

        # Update solution at previous time step.
        self.y_0.x.array[:] = self.y_n.x.array

        self.x = self.y_n.x.array

        return np.copy(self.x), -costs, False, False, {}

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        self.y_0.interpolate(self.initial_condition)
        self.y_n.interpolate(self.initial_condition)
        self.x = self.y_0.x.array

        return np.copy(self.x), {}

    def render(self, mode='human'):
        return

    def close(self):
        return

    def initial_condition(self,x):
        return (x[0]-1)*(x[0]+1)+5

    @property
    def xnorm(self):
        return (self.x**2).sum()