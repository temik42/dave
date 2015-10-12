import pyopencl as cl
import numpy as np


class Smoother():
    def __init__(self, X, sigma, k = 3):
        self.shape = X.shape
        self.sigma = sigma
        self.niter = np.int32(k)
        self.clinit()
        self.loadData(X)
        self.get_params()
        self.Box_sm = self.loadProgram("box_sm.cl")

        
    def clinit(self):
        self.ctx = cl.create_some_context()       
        self.queue = cl.CommandQueue(self.ctx) 

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        kernel_params = {"nx": self.shape[0], "ny": self.shape[1], "nz": self.shape[2],
                         "wx": self.width, "wy": self.width, "wz": self.width, "niter": self.niter}
        return cl.Program(self.ctx, fstr % kernel_params).build()  
        
    def loadData(self, X):
        self._X = X
        self._X1 = np.array(X)
        mf = cl.mem_flags
        self.size = X.nbytes
        
        self.X = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=X)
        self.X1 = cl.Buffer(self.ctx, mf.READ_WRITE, self.size)

        self.queue.finish()
        
    def get_params(self):
        self.width = np.int32(np.floor(0.5*np.sqrt(12./self.niter*self.sigma**2.+1)))
  
    def run(self):        
        self.Box_sm.box_sm_x(self.queue, self.shape, (self.shape[0],1,1), self.X, self.X1)
        cl.enqueue_barrier(self.queue)
        self.Box_sm.box_sm_y(self.queue, self.shape, (1,self.shape[1],1), self.X1, self.X1)
        cl.enqueue_barrier(self.queue)
        self.Box_sm.box_sm_z(self.queue, self.shape, (1,1,self.shape[2]), self.X1, self.X1)
        cl.enqueue_read_buffer(self.queue, self.X1, self._X1)
        self.queue.finish()
        return self
            


     

