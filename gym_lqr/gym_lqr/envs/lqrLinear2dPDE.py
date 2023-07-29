import control
import importlib
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from os import path
import shutil
from numpy.linalg import norm, inv

import dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, plot, io
from dolfinx.fem import petsc

class Linear2dPDEEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, T=2.0, time_steps=200, useExactSolution=False ,useMatCARE=False, Kpath='empty'):
        super(Linear2dPDEEnv, self).__init__()

        # Time discretization -----------------------------------------------------------------------------------------------

        # Define initial and end time value.
        self.T = T

        # Time step configuraton.
        num_steps = time_steps
        self.dt = self.T / num_steps

        #-------------------------------------------------------------------------------------------------------------------

        # Space dicretization.----------------------------------------------------------------------------------------------

        # Two dimensional domain.
        self.Omega = [np.array([0,0]), np.array([1,1])]

        # Two dimensional mesh.
        self.nx = 32
        self.ny = 32
        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, self.nx, self.ny)

        # Resorting index to reshape 1d output vectors.
        mesh_coords = np.transpose(self.domain.geometry.x[:,0:2])
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
  
        self.resortIndex = IY

        # Define test function space.
        self.V = fem.FunctionSpace(self.domain, ("P", 1))

        # Define Dirichlet Boundary Condition. 
        fdim = self.domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(self.domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        self.bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(self.V, fdim, boundary_facets), self.V)

        #-----------------------------------------------------------------------------------------------------------------

        # Iteration variables --------------------------------------------------------------------------------------------
            
        # Create initial condition for the state function and interpolate it in V.

        class Initial_condition():

          def __init__(self):
            pass

          def __call__(self,x):
            values = np.zeros(x.shape,dtype=PETSc.ScalarType)
            values = 3*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
            return values

        self.initial_condition = Initial_condition()

        self.y_0 = fem.Function(self.V)
        self.y_0.name = "y_0"
        self.y_0.interpolate(self.initial_condition)

        self.y_n = fem.Function(self.V)
        self.y_n.name = "y_n"
        self.y_n.interpolate(self.initial_condition)

        self.y_0_exact = fem.Function(self.V)
        self.y_0_exact.name = "y_0"
        self.y_0_exact.interpolate(self.initial_condition)

        self.y_n_exact = fem.Function(self.V)
        self.y_n_exact.name = "y_n"
        self.y_n_exact.interpolate(self.initial_condition)

        #-----------------------------------------------------------------------------------------------------------------

        # Variational Problem (Acctuator functions and coefficients) -----------------------------------------------------

        # PDE coefficients
        self.nu = 0.01

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

        self.a_V = fem.Function(self.V)
        self.a_V.interpolate(PDE_a())

        b_V1 = fem.Function(self.V)
        b_V2 = fem.Function(self.V)
        b_V1.interpolate(PDE_b_1())
        b_V2.interpolate(PDE_b_2())
        self.b_V = ufl.as_vector([b_V1, b_V2])

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

        self.I_w = [fem.Function(self.V) for i in range(0,len(acctuatorDomains))]

        for i, acct in enumerate(acctuatorDomains):
          self.I_w[i].interpolate(lambda x: np.where(np.logical_and(np.logical_and(x[0] - acct[0,0] + tol >= 0, acct[0,1] + tol - x[0] >= 0),np.logical_and(x[1] - acct[1,0] + tol >= 0, acct[1,1] + tol - x[1] >= 0)), 1, 0))

        acctuator = np.array([self.I_w[0].x.array])
        for i in range(1,len(self.I_w)):
          acctuator = np.r_[acctuator, [self.I_w[i].x.array]]
        self.acctuator = acctuator

        #-----------------------------------------------------------------------------------------------------------------

        # Trial and test function.----------------------------------------------------------------------------------------
        y, phi = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        #-----------------------------------------------------------------------------------------------------------------

        # Variational form a_h = <y,phi> + nu*dt*<grad(y),grad(phi)> + dt*<a*y,phi> + dt*<b*y,grad(phi)> -----------------

        self.a_h = y*phi*ufl.dx + self.nu*self.dt*ufl.dot(ufl.grad(y),ufl.grad(phi))*ufl.dx + self.dt*self.a_V*y*phi*ufl.dx - self.dt*ufl.dot(self.b_V*y,ufl.grad(phi))*ufl.dx

        # Define bilinear form.
        self.bilinear_form = fem.form(self.a_h)

        # Assemble FEM matrices.
        self.C = petsc.assemble_matrix(self.bilinear_form)
        self.C.assemble()

        # Create solver.
        solver = PETSc.KSP().create(self.domain.comm)
        solver.setOperators(self.C)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver = solver

        #-----------------------------------------------------------------------------------------------------------------
        
        # RHS to the bilinear form for the exact and approximated cases.--------------------------------------------------

        # Control vector components.
        self.us = [fem.Constant(self.domain, 0.0) for i in range(0,len(self.I_w))]
        self.us_exact = [fem.Constant(self.domain, 0.0) for i in range(0,len(self.I_w))]

        # Variational Problem (LHS).
        self.L = self.us[0]*self.I_w[0]
        for i in range(1,len(self.us)):
          self.L = self.L + self.us[i]*self.I_w[i]
        self.L = (self.y_0 + self.dt*self.L)*phi*ufl.dx

        self.L_exact = self.us_exact[0]*self.I_w[0]
        for i in range(1,len(self.us_exact)):
          self.L_exact = self.L_exact + self.us_exact[i]*self.I_w[i]
        self.L_exact = (self.y_0_exact + self.dt*self.L_exact)*phi*ufl.dx

        # Define linear forms.
        self.linear_form = fem.form(self.L)
        self.linear_form_exact = fem.form(self.L_exact)

        # Assemble LHS vector.
        self.d = petsc.create_vector(self.linear_form)
        self.d_exact = petsc.create_vector(self.linear_form_exact)
        
        #-----------------------------------------------------------------------------------------------------------------

        # FEM and cost functional matrices -------------------------------------------------------------------------------

        # Assembling of matrices for care solver.

        # Mass matrix
        mass_form = y*phi*ufl.dx
        M_dx = petsc.assemble_matrix(fem.form(mass_form), bcs=[self.bc])
        M_dx.assemble()
        M_dx.convert("dense")
        self.M = M_dx.getDenseArray()

        # Stiffness matrix
        stiffnes_form = -1.0*(self.nu*self.dt*ufl.dot(ufl.grad(y),ufl.grad(phi))*ufl.dx + self.dt*self.a_V*y*phi*ufl.dx) + self.dt*ufl.dot(self.b_V*y,ufl.grad(phi))*ufl.dx
        A_dx = petsc.assemble_matrix(fem.form(stiffnes_form), bcs=[self.bc])
        A_dx.assemble()
        A_dx.convert("dense")
        self.A = A_dx.getDenseArray()

        # Stiffness matrix for reward function.
        stiffness_form_reward = ufl.dot(ufl.grad(y),ufl.grad(phi))*ufl.dx
        A_reward_dx = petsc.assemble_matrix(fem.form(stiffness_form_reward), bcs=[self.bc])
        A_reward_dx.assemble()
        A_reward_dx.convert("dense")
        self.A_reward = A_reward_dx.getDenseArray()

        # State dimension
        self.n = self.A.shape[0]

        # Remove boundary
        self.deleteRows = np.where(np.sum(self.M,axis=0)==1)
        self.deleteCols = np.where(np.sum(self.M,axis=1)==1)
        self.A_wb = np.delete(self.A,self.deleteRows,axis=0)
        self.A_wb = np.delete(self.A_wb,self.deleteCols,axis=1)
        self.A_reward_wb = np.delete(self.A_reward,self.deleteRows,axis=0)
        self.A_reward_wb = np.delete(self.A_reward_wb,self.deleteCols,axis=1)
        self.M_wb = np.delete(self.M,self.deleteRows,axis=0)
        self.M_wb = np.delete(self.M_wb,self.deleteCols,axis=1)

        # Acctuator matrix
        self.Ac = np.array([self.I_w[0].x.array])
        for i in range(1,len(self.I_w)):
          self.Ac = np.r_[self.Ac, [self.I_w[i].x.array]]
        self.Ac = self.Ac.T
        self.Ac = np.delete(self.Ac,self.deleteRows,axis=0)
        self.B = np.matmul(self.M_wb,self.Ac)

        # State dimension without boundary
        self.n_wb = self.A_wb.shape[0]

        # Action dimesion
        self.m = self.B.shape[1]

        # Matrices and coefficients for the reward function r = alpha*y'*Q*y + beta*u'*R*u
        self.alpha = 1.
        self.beta = 1.
        self.Q = (self.alpha/2)*self.A_reward
        self.Q_wb = (self.alpha/2)*self.A_reward_wb
        self.R = (self.beta/2)*np.eye(self.m)

        #-----------------------------------------------------------------------------------------------------------------

        # CARE solution for exact solution in the linear pde case --------------------------------------------------------

        self.useExactSolution = useExactSolution  
        self.useMatCARE = useMatCARE
        self.Kpath = Kpath
        self.K = np.zeros((self.m, self.n_wb))

        if self.useMatCARE:
           if self.Kpath!='empty':
              self.K = np.load(path.join(path.dirname(__file__), "data/Ricatti_solution_matrices/{}.npy".format(Kpath)))
           else:
              self.K = np.zeros((self.m, self.n_wb))
        else:
           # Continuous discount factor.
           gamma = .99999
           conti_gamma = (1 - gamma) / self.dt

           # Solve discounted generalized CARE
           Ad = self.A_wb - (conti_gamma / 2.) * np.eye(self.n_wb)
           X, L, K = control.care(Ad, self.B, self.Q_wb, self.R, np.zeros((self.n_wb, self.m)), self.M_wb)
           self.K = np.asarray(K)

        #-----------------------------------------------------------------------------------------------------------------

        # Initial control.------------------------------------------------------------------------------------------------

        self.initial_action = np.zeros((len(acctuatorDomains),))

        # Action and state spaces ----------------------------------------------------------------------------------------

        # Random sampling from the action space : $U[-1, 1)$
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

        # State solution variables 
        self.x = None
        self.exact_x = None

    def get_exact_action(self, x):
        u = -self.K @ np.delete(x,self.deleteRows,axis=0)
        return u

    def exact_step(self, u):

        # Quadratic cost function x^T Q x + u^T R u
        costs = np.sum(np.dot(np.matmul(np.transpose(self.exact_x),self.Q),self.exact_x)) + np.sum(np.dot(np.matmul(np.transpose(u),self.R),u))

        # Update coefficients of control vector.
        uls = u.tolist()
        for i in range(0,len(self.us_exact)):
          self.us_exact[i].value = uls[i]

        # Update the right hand side reusing the initial vector
        with self.d_exact.localForm() as loc_b:
             loc_b.set(0)
        petsc.assemble_vector(self.d_exact, self.linear_form_exact)

        # Apply Dirichlet boundary condition to the vector
        petsc.apply_lifting(self.d_exact, [self.bilinear_form], [[self.bc]])
        self.d_exact.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(self.d_exact, [self.bc])

        # Solve linear problem
        self.solver.solve(self.d_exact, self.y_n_exact.vector)
        self.y_n_exact.x.scatter_forward()

        # Update solution at previous time step
        self.y_0_exact.x.array[:] = self.y_n_exact.x.array

        self.exact_x = self.y_n_exact.x.array

        return np.copy(self.exact_x), -costs, False, False, {}

    def step(self, u):

        # Quadratic cost function x^T Q x + u^T R u
        costs = np.sum(np.dot(np.matmul(np.transpose(self.x),self.Q),self.x)) + np.sum(np.dot(np.matmul(np.transpose(u),self.R),u))

        # Update coefficients of control vector.
        uls = u.tolist()
        for i in range(0,len(self.us)):
          self.us[i].value = uls[i]

        # Update the right hand side reusing the initial vector
        with self.d.localForm() as loc_b:
             loc_b.set(0)
        petsc.assemble_vector(self.d, self.linear_form)

        # Apply Dirichlet boundary condition to the vector
        petsc.apply_lifting(self.d, [self.bilinear_form], [[self.bc]])
        self.d.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(self.d, [self.bc])

        # Solve linear problem
        self.solver.solve(self.d, self.y_n.vector)
        self.y_n.x.scatter_forward()

        # Update solution at previous time step
        self.y_0.x.array[:] = self.y_n.x.array

        self.x = self.y_n.x.array

        return np.copy(self.x), -costs, False, False, {}

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        self.y_0 = fem.Function(self.V)
        self.y_0.name = "y_0"
        self.y_0.interpolate(self.initial_condition)

        self.y_n = fem.Function(self.V)
        self.y_n.name = "y_n"
        self.y_n.interpolate(self.initial_condition)

        self.y_0_exact = fem.Function(self.V)
        self.y_0_exact.name = "y_0"
        self.y_0_exact.interpolate(self.initial_condition)

        self.y_n_exact = fem.Function(self.V)
        self.y_n_exact.name = "y_n"
        self.y_n_exact.interpolate(self.initial_condition)

        self.x = self.y_0.x.array
        self.exact_x = self.x

        return np.copy(self.x), {}

    def render(self, mode='human'):
        return

    def close(self):
        return

    @property
    def xnorm(self):
        return (self.x**2).sum()
