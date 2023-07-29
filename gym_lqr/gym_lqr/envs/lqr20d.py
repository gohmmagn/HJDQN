import numpy as np
import control
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from os import path
from numpy.linalg import norm


class LinearQuadraticRegulator20DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, useExactSolution=False, useMatCARE=False, Kpath="empty"):
        # simulate LQR until $T = 10$
        self.dt = .05
        self.n = 20
        self.A = np.load(path.join(path.dirname(__file__), "data/A20.npy"))
        self.B = np.load(path.join(path.dirname(__file__), "data/B20.npy"))

        Q = 5. * np.eye(self.n)
        R = np.eye(self.n)
        h = self.dt

        self.useExactSolution = useExactSolution
        self.useMatCARE = useMatCARE
        self.Kpath = Kpath

        if self.useMatCARE:
           if self.Kpath!='empty':
              self.K = np.load(path.join(path.dirname(__file__), "data/Ricatti_solution_matrices/{}.npy".format(Kpath)))
           else:
              self.K = np.zeros(self.n)
        else:
           # Continuous discount factor.
           gamma = .99999
           conti_gamma = (1 - gamma) / h

           # Solve discounted generalized CARE
           Ad = self.A - (conti_gamma / 2.) * np.eye(self.n)
           X, L, K = control.lqr(Ad, self.B, Q, R)
           self.K = np.asarray(K)

        # random sampling from the action space : $U[-1, 1)$
        self.action_space = spaces.Box(
            low=-1.,
            high=1., shape=(self.n,),
            dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, shape=(self.n,),
            dtype=np.float64
        )
        self.x = None
        self.exact_x = None

    def get_exact_action(self, x):
        u = -self.K @ x
        return u

    def exact_step(self, u):
        h = self.dt
        # quadratic cost function x^T Q x + u^T R u
        costs = 5. * np.sum(self.exact_x**2) + np.sum(u**2)  # Q = 5I, R = I

        def ftn(t, y):
            # nested function which is determined by free variables A, B, and u
            # note that our dynamical system is time-homogeneous
            return self.A @ y + self.B @ u

        # Runge-Kutta 4-th order method
        k1 = ftn(.0, self.exact_x)
        k2 = ftn(.0 + .5 * h, self.exact_x + .5 * h * k1)
        k3 = ftn(.0 + .5 * h, self.exact_x + .5 * h * k2)
        k4 = ftn(.0 + h, self.exact_x + h * k3)

        dx = h * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        self.exact_x = self.exact_x + dx                    # x(t + h) = x(t) + \int_t^{t+h} \dot x dt
        return np.copy(self.exact_x), -costs, False, False, {}	

    def step(self, u):
        h = self.dt
        # quadratic cost function x^T Q x + u^T R u
        costs = 5. * np.sum(self.x**2) + np.sum(u**2)  # Q = 5I, R = I

        def ftn(t, y):
            # nested function which is determined by free variables A, B, and u
            # note that our dynamical system is time-homogeneous
            return self.A @ y + self.B @ u

        # Runge-Kutta 4-th order method
        k1 = ftn(.0, self.x)
        k2 = ftn(.0 + .5 * h, self.x + .5 * h * k1)
        k3 = ftn(.0 + .5 * h, self.x + .5 * h * k2)
        k4 = ftn(.0 + h, self.x + h * k3)

        dx = h * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        self.x = self.x + dx                    # x(t + h) = x(t) + \int_t^{t+h} \dot x dt
        return np.copy(self.x), -costs, False, False, {}

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        # sample the initial state vector uniformly from $U[-1, 1)$
        self.x = 2. * np.random.rand(self.n) - 1.
        self.exact_x = self.x
        return np.copy(self.x), {}

    def close(self):
        return

    def render(self, mode='human'):
        return

    @property
    def xnorm(self):
        return (self.x**2).sum()**.5

    #def set_state(self, x1):
    #    self.x = x1
