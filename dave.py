import numpy as np
from lusol import Solver
from scipy.ndimage.filters import correlate1d

class DAVE():
    def __init__(self, X, sigma = (3,3,3)):
        self.sigma = sigma
        self.X = X.astype(np.float32)
        self.dim = self.X.shape
        
        h = np.floor(sigma).astype(np.int32)
        self.x = [np.arange(-4*h[i],4*h[i]+1, dtype = np.float32) for i in range(0,3)]
        self.exp = [np.exp(-self.x[i]**2./2/sigma[i]**2.) for i in range(0,3)]

    def convol(self, X, mul = (0,0,0)):
        out = np.array(X).astype(np.float32)
        for i in range(0,3):
            if (mul[i] == 0):
                correlate1d(out, self.exp[i], axis=i, output = out)
            else:
                correlate1d(out, self.exp[i]*self.x[i]**mul[i], axis=i, output = out)
        return out
    
    def run(self):
        G = np.gradient(self.X)

        A = np.zeros((6,6)+self.dim, dtype = np.float32)
        b = np.zeros((6,)+self.dim, dtype = np.float32)

        A[0,0,:,:,:] = self.convol(G[0]*G[0])
        A[1,0,:,:,:] = self.convol(G[0]*G[1])
        A[1,1,:,:,:] = self.convol(G[1]*G[1])
        A[2,0,:,:,:] = self.convol(self.X*G[0]) + self.convol(G[0]*G[0], (1,0,0))
        A[2,1,:,:,:] = self.convol(self.X*G[1]) + self.convol(G[0]*G[1], (1,0,0))
        A[2,2,:,:,:] = self.convol(self.X**2) + 2*self.convol(self.X*G[0], (1,0,0)) + self.convol(G[0]*G[0], (2,0,0))
        A[3,0,:,:,:] = self.convol(self.X*G[0]) + self.convol(G[0]*G[1], (0,1,0))
        A[3,1,:,:,:] = self.convol(self.X*G[1]) + self.convol(G[1]*G[1], (0,1,0))
        A[3,2,:,:,:] = (self.convol(self.X**2) + self.convol(self.X*G[0], (1,0,0))
                        + self.convol(self.X*G[1], (0,1,0)) + self.convol(G[0]*G[1], (1,1,0)))
        A[3,3,:,:,:] = self.convol(self.X**2) + 2*self.convol(self.X*G[1], (0,1,0)) + self.convol(G[1]*G[1], (0,2,0))
        A[4,0,:,:,:] = self.convol(G[0]*G[0], (0,1,0))
        A[4,1,:,:,:] = self.convol(G[0]*G[1], (0,1,0))
        A[4,2,:,:,:] = self.convol(self.X*G[0], (0,1,0)) + self.convol(G[0]*G[0], (1,1,0))
        A[4,3,:,:,:] = self.convol(self.X*G[0], (0,1,0)) + self.convol(G[0]*G[1], (0,2,0))
        A[4,4,:,:,:] = self.convol(G[0]*G[0], (0,2,0))
        A[5,0,:,:,:] = self.convol(G[0]*G[1], (1,0,0))
        A[5,1,:,:,:] = self.convol(G[1]*G[1], (1,0,0))
        A[5,2,:,:,:] = self.convol(self.X*G[1], (1,0,0)) + self.convol(G[0]*G[1], (2,0,0))
        A[5,3,:,:,:] = self.convol(self.X*G[1], (1,0,0)) + self.convol(G[1]*G[1], (1,1,0))
        A[5,4,:,:,:] = self.convol(G[0]*G[1], (1,1,0))
        A[5,5,:,:,:] = self.convol(G[1]*G[1], (0,2,0))

        b[0,:,:,:] = self.convol(G[0]*G[2])
        b[1,:,:,:] = self.convol(G[1]*G[2])
        b[2,:,:,:] = self.convol(self.X*G[2]) + self.convol(G[0]*G[2], (1,0,0))
        b[3,:,:,:] = self.convol(self.X*G[2]) + self.convol(G[1]*G[2], (0,1,0))
        b[4,:,:,:] = self.convol(G[0]*G[2], (0,1,0))
        b[5,:,:,:] = self.convol(G[1]*G[2], (1,0,0))

        for i in range(1,6):
            for j in range(0,i):
                A[j,i,:,:,:] = A[i,j,:,:,:]
                
        self.cle = Solver(np.array(A, dtype = np.float32),np.array(b, dtype = np.float32)).run()    
        return self
    
    def get(self):
        return self.cle.get()