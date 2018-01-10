import numpy as np
import matplotlib.pyplot as plt

import GaussLobattoPoly
import controlHamiltonian

from scipy.optimize import minimize


class ControlSys:
    def __init__(
            self,
            order,
            state_dims,
            control_dims,
            start_time,
            end_time,
            num_fine_t = None,
            ps_poly=None,
            controlHam=None
    ):
        self.num_x_dims = state_dims
        self.num_u_dims = control_dims
        self.num_t_nodes = order
        self.path_length = 1.0
        self.total_size = (self.num_x_dims + self.num_u_dims)*(self.num_t_nodes + 1) + 2
        self.x_vec = np.array(   self.total_size, dtype=float32)


        if ps_poly is None:
            ps_poly = GaussLobattoPoly(order)
        self.ps_poly = ps_poly

        self.start_time = start_time
        self.end_time = end_time
        h = start_time - end_time
        if controlHam is None:
            controlHam = controlHamiltonian(h)
        self.controlHam = controlHam

        if num_fine_t is None:
            num_fine_t = 1000
        self.num_fine_t = num_fine_t

        x_size_idx = (self.num_x_dims)*(self.num_t_nodes + 1)
        self.x_states_idx = range(0,x_size_idx)
        self.u_controls_idx = range(x_size_idx,  (self.num_x_dims + self.num_u_dims)*(self.num_t_nodes + 1))
        #self.end_idx = (self.num_x_dims + self.num_u_dims)*(self.num_t_nodes + 1)
        self.end_idx = range(x_size_idx, x_size_idx + 2*(self.num_x_dims))

    def vecToState(self):
        x_state = self.x_vec[self.x_states_idx]
        x_state = x_state.reshape(((self.num_t_nodes + 1), (self.num_x_dims) ))
        return x_state

    def vecToState(self, x):
        x_state = x[self.x_states_idx]
        x_state = x_state.reshape(((self.num_t_nodes + 1), (self.num_x_dims) ))
        return x_state

    def vecToControl(self):
        u_control = self.x_vec[self.u_controls_idx]
        u_control = u_control.reshape(((self.num_t_nodes + 1), (self.num_u_dims) ))
        return u_control

    def vecToControl(self, x):
        u_control = x[self.u_controls_idx]
        u_control = u_control.reshape(((self.num_t_nodes + 1), (self.num_u_dims) ))
        return u_control

    def stateDimSelect(self, idx):
        x_states_idx_temp = range((idx) * (self.num_t_nodes + 1), (idx+1) * (self.num_t_nodes + 1) )
        return self.x_vec[x_states_idx_temp ]

    # def matrixToVec(self, x):
    #     #x_vec_temp = np.array(1, self.total_size, dtype=float32)
    #     x_vec_temp = x.flatten()
    #
    #     return x_vec_temp


    def evalDotX(self, x):
        #xi = np.array((self.num_t_nodes + 1, self.num_x_dims), dtype=float32)
        D = self.ps_poly.Dmatrix
        x_state = self.vecToState(x);
        u_controls = self.vecToControl(x);
        xi = np.dot(D, x_state)

        yi = self.controlHam.dHdp(x_state, u_controls)
        zi = xi - yi

        zi = zi.flatten('F')
        return zi

    def evalBoundary(self, x):
        x_state = self.vecToState(x);
        x_state_init = x_state[0,:]
        x_state_end = x_state[-1,:]

        zi_init = x_state_init - self.controlHam.dHdp0()
        zi_init = zi_init.flatten('F')
        zi_end = x_state_end  - self.controlHam.dHdpT()
        zi_end = zi_end.flatten('F')

        zi = np.append(zi_init, zi_end)

        return zi

    def constraints(self, x):
        f = np.zeros((self.num_x_dims)*(self.num_t_nodes+1) + (self.num_x_dims)*(2))
        f[self.x_states_idx] = self.evalDotX()
        f[self.end_idx] = self.evalBoundary()

        return f

    def inequalityConstrait(self,x):
        x_state = self.vecToState(x);
        x_path = x_state[:, 0]
        x_path = x_path - self.path_length
        return x_path


    def solveOC(self):
        #run fmincond
        #initial guess
        x0 = np.zeros((self.num_x_dims + self.num_u_dims) * (self.num_t_nodes + 1) )

        #using SciPy minimize function
        eqCond = {'type': 'eq', 'fun': self.constraints}
        inCond = {'type': 'ineq', 'fun': self.inequalityConstrait}
        cons = ([eqCond, inCond])
        solution = minimize(objective, x0, method='SLSQP', bounds=None, constraints=cons)
        self.x_vec = solution.x
        return solution.x




    def reconstructStates(self):
        #Convert from pseudospectral coefficients to functions
        t_fine = np.linspace(self.start_time, self.end_time, self.num_fine_t)
        t_scaled = self.ps_poly.scaleVec(self.start_time, self.end_time)
        x_reconst = np.zeros( (self.num_fine_t, self.num_x_dims))
        x_state = self.vecToState()

        for i in range(self.num_x_dims):
            x_reconst[:,i] = self.ps_poly.lgl_interpolate(t_scaled , x_state[:,i],t_fine)

        return x_reconst, t_fine , x_state, t_scaled


    def printResults(self):
        #use matplotlib

        x_reconst, t_fine,  x_state, t_scaled = self.reconstructStates(self)
        plt.subplot(2,1,1)
        plt.plot(t_fine, x_reconst[:,0], '-')
        plt.title('Pseudospectral Method - States')
        plt.ylabel('Position')

        plt.subplot(2, 1, 2)
        plt.plot(t_fine, x_reconst[:, 1], '-')
        plt.xlabel('time (s)')
        plt.ylabel('Velocity')

        plt.subplot(2,1,1)
        plt.plot(t_scaled,  x_state[:,0], 'o')

        plt.subplot(2,1,2)
        plt.plot(t_scaled,  x_state[:,1], 'o')
        ax = plt.gca()
        ax.set_xticklabels([])

        plt.show()
