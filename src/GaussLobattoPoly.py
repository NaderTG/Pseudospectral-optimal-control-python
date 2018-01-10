import numpy as np

class GaussLobattoPoly:
    def __init__(self, order, tol = 0.000001):
        self.order = order -1
        self.x_nodes = None
        self.weights = None
        self.Dmatrix = None
        self.tol  = tol
        self.lgl_nodes()
        self.lgl_matrix()


    def lgl_nodes(self):
        N1 = self.order + 1
        x = np.cos(np.linspace(0, np.pi, N1))
        p = np.zeros((N1, N1))
        err = 1

        while err > self.tol :
            xold = x
            p[:, 0] = 1;
            p[:, 1] = x;

            for k in range(2, N1):
                p[:, k] = ( (2*k-1)*x*p[:,k-1] - (k-1)*p[:,k-2]) / (k);

            x = xold - (x*p[:, N1-1] - p[:,N1-2])/(N1*p[:, N1-1]);
            err_temp = np.absolute(x - xold)
            err = err_temp.max() ;

        for i in range(0, N1/2):
            temp = x[N1 - i -1]
            x[N1 - i -1] = x[i ]
            x[i] = temp


        self.x_nodes = x;
        self.weights = 2.0/(self.order * N1 * p[:, N1-1]* p[:, N1-1]);

    def legpoly(self, idx):
        pn = 1;
        pn_p1 = 1;
        x = self.x_nodes[idx]
        N = self.order

        if N > 0:
            pn_1 = x * pn
            if N == 1:
                pn = pn_1
            for j in range(2, N + 1):
                pn_p1 = (x * (2 * j - 1) * pn_1 - (j - 1) * pn) / j
                pn = pn_1
                pn_1 = pn_p1
            pn = pn_p1

        return pn

    def lgl_matrix(self):
        N1 = self.order + 1
        matrixD = np.zeros((N1, N1))

        for i in range(0, N1):
            for j in range(0, N1):
                if (i != j):

                    Li = self.legpoly(i)
                    Lj = self.legpoly(j)
                    matrixD[i,j] = (Li/Lj)*(1.0/(self.x_nodes[i] - self.x_nodes[j]))

        matrixD[0,0] = - (N1*self.order )/4.0
        matrixD[N1-1, N1-1] = (N1*self.order )/4.0
        self.Dmatrix = matrixD

    def lgl_interpolate(self, x, y,xi):
        n = x.size - 1
        ni = xi.size
        L = np.ones((n + 1, ni), dtype=float);

        for k in range(0,n + 1):
            for kk in range(0, k ):
                L[kk , :] = L[kk , :] * ((xi - x[k ]) / (x[kk ] - x[k ]))

            for kk in range(k + 1, n+1):
                L[kk , :] = L[kk , :] * ((xi - x[k ]) / (x[kk ] - x[k ]))

        return np.dot(y,L)

    def scaleVec(self, t0, tf):
        t = self.x_nodes*(tf - t0) + (tf + t0)/2.0
        return t
