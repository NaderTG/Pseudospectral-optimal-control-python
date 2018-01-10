import numpy as np

class controlHamiltonian:
    def __init__(self, h):
        self.h = h

    def l(self, u):
        sum = 0
        size_u_state = u.size
        for i in range(size_u_state):
            sum = sum + 0.5 * u[i] * u[i]

        return sum

    def dHdp(self, x,u):
        F = np.zeros(x.shape )
        F[:,0] = x[:,1]
        F[:, 1] = u[:,0]
        F = (self.h/2.0)*F
        return F

    def dHdp0(self):
        #This is a problem specific implementation
        x_init = np.zeros(2)
        x_init[0] = 0
        x_init[1] = 1
        return x_init

    def dHdpT(self):
        x_end = np.zeros(2)
        x_end[0] = 0
        x_end[1] = -1
        return x_end
