import numpy as np

#import IonModel


class BidomainSolver:
    '''BidomainSolver class for solving the bidomain equations using the Crank-Nicolson method
    
    Parameters:
    dimension (int): Dimension of the problem (default is 1). Will expand to 2D and 3D in the future
    parallelization (bool): Parallelization flag (default is False)
    ionModel (str): Not yet implemented.
    time (float): Total simulation time (default is 0.05)
    timestep_size (float): Timestep size (default is 0.001)
    shape (tuple): Shape of the grid (default is (100,))
    pacing (str): Pacing protocol (default is None). Will allow time-dependent external current in the future
    conductivity (float): Conductivity of the tissue (default is 1.0). Sets the maximum conductivity of the tissue
    
    To-Do:
    - Expand the Heat Equation to the full Reaction-Diffusion model
    - Build the IonModel class, start with minimal Hodgkin-Huxley model
    - Introduce the external potential and couple to the transmembrane potential
    - Implement the full bidomain model with the two equations'''

    def __init__(self, dimension=1, parallelization=False, ionModel='HH', time=0.05, timestep_size=0.001, shape=(100,), pacing=None, conductivity=1.0):
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
        self.S = np.ones(self.shape)

    def initialize_v(self,IC='uniform'):
        # Initialize the voltage field v_tm
        self.N = int(len(self.shape) / self.dx)
        self.v_tm = np.ones(self.N)

        if IC == 'uniform': # Very boring, mostly for testing
            pass
        elif IC == 'step':
            self.v_tm[int(self.N/2):] = 2*self.v_tm[int(self.N/2):]
        elif IC == 'sine':
            self.v_tm = 1 + 0.5*np.sin(np.pi*np.linspace(0, self.L, self.N))
        else:
            raise ValueError('Initial Condition not supported')

    def cn_init_1d(self,BC,het_input='homogeneous'):
        # Initialize the Crank-Nicolson method for a 1D Scalar Field
        
        N = self.N # Number of grid points
        S = self.S # Conductivity of each grid point
        S_max = self.conductivity # Maximum conductivity

        # Optional heterogeneity in the conductivity
        if het_input == 'step':
            S = S_max*np.ones(N)
            S[int(N/2):] = 0.1*S_max
        elif het_input == 'random':
            S = S_max*np.random.rand(N)
        elif het_input == 'homogeneous':
            S = S_max*np.ones(N)
        else:
            raise ValueError('Heterogeneity not supported')

        alpha = S*self.dt/(2*self.dx**2) # Constants

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
        if BC == 'Dirichlet': # Fixed value at the boundaries
            A[0,0] = 1
            A[-1,-1] = 1
            B[0,0] = 1
            B[-1,-1] = 1
        elif BC == 'Neumann': # Zero derivative at the boundaries
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
        
        # Gaussian elimination
        for i in range(1,n):
            factor = matrix[i,i-1] / matrix[i-1,i-1]
            matrix[i,i] -= factor * matrix[i-1,i]
            rhs[i] -= factor * rhs[i-1]
        x[-1] = rhs[-1] / matrix[-1,-1] # Last element
        
        # Backward substitution
        for i in range(n-2,-1,-1):
            x[i] = (rhs[i] - matrix[i,i+1] * x[i+1]) / matrix[i,i]
            
        # Update the solution
        self.v_tm = np.copy(x)

    def cn_solve(self):
        ## dv_tm/dt = S * d^2v_tm/dx^2 - ion_current(v_tm,)/C_m
        ## v_tm(x,t+dt) = v_tm(x,t) + dt * (S * d^2v_tm/dx^2 - ion_current(v_tm,)/C_m)
        ## A * v_tp = B * v_tm (v_tp is the predicted solution at t+dt, v_tm is the solution at t)
        ## A is a tri-diagonal matrix with 1 + 2*alpha on the diagonal and -alpha on the off-diagonals
        ## B is a tri-diagonal matrix with 1 - 2*alpha on the diagonal and alpha on the off-diagonals

        # Matrix Solver - Crank-Nicolson Method
        while self.t < self.time:
            b = self.B @ self.v_tm # multiply the right-hand side
            self.thomas_method(b) # use the Thomas Algorithm to solve the system
            self.t += self.dt

    def apply_pacing(self):
        # This will eventually add an external source, function of time
        pass



