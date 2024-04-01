import numpy as np

#import IonModel


class BidomainSolver:
    def __init__(self, dimension=1, parallelization=False, ionModel='HH', time=1.0, timestep_size=0.01, shape=(100,), pacing=None, conductivity=1.0):
        self.dimension = dimension
        self.parallelization = parallelization
        #self.ionModel = ionModel if ionModel else IonModel() # IonModel class to be implemented
        self.time = time
        self.dt = timestep_size
        self.shape = shape
        self.pacing = pacing
        self.conductivity = conductivity
        self.dx = 0.01
        self.t = 0.0

    def initialize_v(self,IC='uniform'):
        self.N = int(len(self.shape) / self.dx)
        self.v_tm = np.ones(self.N)

        if IC == 'uniform':
            pass
        elif IC == 'step':
            self.v_tm[int(self.N/2):] = 2*self.v_tm[int(self.N/2):]
        elif IC == 'sine':
            self.v_tm = 1 + 0.5*np.sin(np.pi*np.linspace(0, self.L, self.N))
        else:
            raise ValueError('Initial Condition not supported')


    ## Crank-Nicolson initialization
    def cn_init_1d(self,S_input,BC,het_input='homogeneous'):
        # Initialize the Crank-Nicolson method for a 1D problem
        N = self.N

        # Optional heterogeneity in the conductivity
        if het_input == 'step':
            S = np.ones(N)
            S[int(N/2):] = 0.1
        elif het_input == 'sine':
            S = 1 + 0.5*np.sin(np.pi*np.linspace(0, self.L, N))
        elif het_input == 'random':
            S = S_input*np.random.rand(N)
        elif het_input == 'homogeneous':
            S = S_input*np.ones(N)
        else:
            raise ValueError('Heterogeneity not supported')

        alpha = S*self.dt/(2*self.dx**2)

        # Matrix A
        A = np.zeros((N,N))
        for i in range(1,N-1):
            A[i,i] = 1 + 2*alpha[i]
            A[i,i-1] = -alpha[i]
            A[i,i+1] = -alpha[i]

        # Matrix B
        B = np.zeros((N,self.N))
        for i in range(1,N-1):
            B[i,i] = 1 - 2*alpha[i]
            B[i,i-1] = alpha[i]
            B[i,i+1] = alpha[i]

        # Boundary conditions
        if BC == 'Dirichlet':
            A[0,0] = 1
            A[-1,-1] = 1
            B[0,0] = 1
            B[-1,-1] = 1
        elif BC == 'Neumann':
            A[0,0] = 1 + alpha[0]
            A[-1,-1] = 1 + alpha[-1]
            A[0,1] = -alpha[0]
            A[-1,-2] = -alpha[-1]
            B[0,0] = 1 - alpha[0]
            B[-1,-1] = 1 - alpha[-1]
            B[0,1] = alpha[0]
            B[-1,-2] = alpha[-1]
        else:
            raise ValueError('Boundary condition not supported')
        
        self.A,self.B = A,B

    def thomas_method(self, Bv_tm):
        # Check if the matrix is tridiagonal
        if not np.all(self.A == np.diag(np.diag(self.A)) + 
                      np.diag(np.diag(self.A, k=1), k=1) + 
                      np.diag(np.diag(self.A, k=-1), k=-1)):
            raise ValueError("Input matrix is not tridiagonal")
        
        n = len(self.A)
        x = np.zeros(n)
        rhs = Bv_tm

        matrix = self.A.copy() # Copy the matrix to avoid modifying the original
        
        # Perform Gaussian elimination
        for i in range(1, n):
            m = matrix[i][i-1] / matrix[i-1][i-1]
            matrix[i] -= m * matrix[i-1]
            rhs[i] -= m * rhs[i-1]
        
        # Perform backsubstitution
        x[n-1] = rhs[n-1] / matrix[n-1][n-1]
        for i in range(n-2, -1, -1):
            x[i] = ([i] - matrix[i][i+1] * x[i+1]) / matrix[i][i]
        
        del matrix
        self.v_tm = np.copy(x)

    def cn_solve(self):
        ## dv_tm/dt = S * d^2v_tm/dx^2 - ion_current(v_tm,)/C_m
        ## v_tm(x,t+dt) = v_tm(x,t) + dt * (S * d^2v_tm/dx^2 - ion_current(v_tm,)/C_m)
        ## A * v_tp = B * v_tm
        ## A is a tri-diagonal matrix with 1 + 2*alpha on the diagonal and -alpha on the off-diagonals
        ## B is a tri-diagonal matrix with 1 - 2*alpha on the diagonal and alpha on the off-diagonals


        while self.t < self.time:
            rhs = self.B @ self.v_tm # This will also include the ion current term eventually
            self.thomas_method(rhs)
            self.t += self.dt



    def apply_pacing(self):
        # This will eventually apply a pacing protocol
        pass

    def solve(self):
        # Main solver method implementing the Crank-Nicolson method
        pass

    def run(self):
        # High-level method to run the solver for the specified time
        # Calls solve() method in a loop with time-stepping
        pass


