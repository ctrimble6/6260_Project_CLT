""" Bidomain Module
    This is the under-development copy of my Bidomain Solver.
    See bidomainCLT.py for the current working copy."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Bidomain():
    def __init__(self,num_cells,dt = 0.001,cell_model = 'BR',conductivity_profile = 'uniform',s_factor =0.2,IC='uniform',stimulation='step',boundary='periodic') -> None:

        self.num_cells = num_cells

        # Initialize the Bidomain Variables
        self.max_conductivity = 0.17
        self.sV_ratio = 1.0
        self.V0 = -84.0
        self.Cm = 1.0
        self.dt = dt
        self.dx = 0.01
        self.time = 0.0
        self.tmax = 600.0
        self.boundary = boundary
        self.IC = IC
        self.stimulation = stimulation

        self.dtCm = self.dt/self.Cm
        self.tcount = 0

        # Initialize the Conductivity
        self.conductivity_profile = conductivity_profile
        self.s_factor = s_factor
        self.si = np.atleast_1d(np.ones(self.num_cells))
        self.se = np.atleast_1d(np.ones(self.num_cells))
        if self.conductivity_profile == 'uniform':
            pass
        elif self.conductivity_profile == 'step':
            self.si[int(self.num_cells/2):] = self.s_factor*self.si[int(self.num_cells/2):]
            self.se[int(self.num_cells/2):] = 1
        elif self.conductivity_profile == 'square':
            self.si[int(self.num_cells/4):int(3*self.num_cells/4)] = self.s_factor*self.si[int(self.num_cells/4):int(3*self.num_cells/4)]
            self.se[int(self.num_cells/4):int(3*self.num_cells/4)] = 1

        self.sigma = self.max_conductivity*np.ones(self.num_cells)*(self.si/(self.si + self.se))


        # Initialize the Potential (MUST start at rest potential!)
        self.Vm = np.zeros((self.num_cells,2)) # Deprecated: switched to using self.voltage
        self.Vm[:,0] = self.V0 + 0.01*np.random.randn(self.num_cells)
        self.Vm[:,1] = self.Vm[:,0]


        # Initialize the Cell Model
        self.cell_model = cell_model

        if self.cell_model == 'BR': # Beeler-Reuter 
            import cell_module_BR as cell_module
            cell = cell_module.Cell_BR(self.Vm[:,0],dt = self.dt,num_cells = self.num_cells)
        else: # Could add more cell models here
            print('Cell model not found')
            return
        
        self.cell = cell

        # Initialize Crank-Nicolson Method
        self.alpha = self.sigma*self.dt/(2*self.dx**2)
        self.A = np.zeros((self.num_cells,self.num_cells))
        self.B = np.zeros((self.num_cells,self.num_cells))

        for i in range(1,self.num_cells-1):
            self.A[i,i] = 1 + 2*self.alpha[i]
            self.A[i,i-1] = -self.alpha[i]
            self.A[i,i+1] = -self.alpha[i]

            self.B[i,i] = 1 - 2*self.alpha[i]
            self.B[i,i-1] = self.alpha[i]
            self.B[i,i+1] = self.alpha[i]

        if self.boundary == 'Dirichlet':
            self.A[0,0] = 1
            self.A[-1,-1] = 1
            self.B[0,0] = 1
            self.B[-1,-1] = 1
        elif self.boundary == 'Neumann':
            self.A[0,0] = 1 + self.alpha[0]
            self.A[-1,-1] = 1 + self.alpha[-1]
            self.A[0,1] = -self.alpha[0]
            self.A[-1,-2] = -self.alpha[-1]
            self.B[0,0] = 1 - self.alpha[0]
            self.B[-1,-1] = 1 - self.alpha[-1]
            self.B[0,1] = self.alpha[0]
            self.B[-1,-2] = self.alpha[-1]
        else:
            print('Boundary condition not supported')
            return
        
    def initialize_voltage(self):
        n_cells = self.num_cells
        n_time = int(self.tmax/self.dt)

        # Create a 2D array to store the voltage of each cell at each time step
        self.voltage = np.zeros((n_cells, n_time))
        self.voltage[:, 0] = np.copy(self.Vm[:, 0])
        

    def thomas_method(self):
        # Simple Gaussian elimination for tridiagonal matrices
        matrix = np.copy(self.A)
        if not np.all(matrix == np.diag(np.diag(matrix)) + 
                      np.diag(np.diag(matrix, k=1), k=1) + 
                      np.diag(np.diag(matrix, k=-1), k=-1)):
            raise ValueError("Input matrix is not tridiagonal")
        
        N = self.num_cells
        vm_New = np.zeros(N)
        d = np.zeros(N)
        v_copy = np.copy(self.voltage[:,self.tcount])

        d = self.B @ v_copy

        # Gaussian elimination
        for i in range(1,N):
            factor = matrix[i,i-1] / matrix[i-1,i-1]
            matrix[i,i] -= factor * matrix[i-1,i]
            d[i] -= factor * d[i-1]
        vm_New[-1] = d[-1] / matrix[-1,-1]

        # Backward substitution
        for i in range(N-2,-1,-1):
            vm_New[i] = (d[i] - matrix[i,i+1] * vm_New[i+1]) / matrix[i,i]
        try:
            self.voltage[:,self.tcount+1] = vm_New
        except IndexError:
            pass

    def update(self):
        
        # Switch to using self.voltage instead of self.Vm
        
        if self.tcount >= len(self.voltage[0])-1:
            self.run = False
            print('Simulation complete')
        
        # Update the cell state
        self.cell.update(self.voltage[:,self.tcount])
        self.voltage[:,self.tcount] = self.voltage[:,self.tcount] - self.dtCm*(self.cell.I_total[:])
        
        # Stimulus -- Still hardcoded for now
        if 10 < self.time > 10.3:
            self.voltage[0:12,self.tcount] += 10*np.exp(-((self.time-10.3)**2)/0.01)

        self.thomas_method()
        self.tcount += 1
        
        self.time += self.dt


    def run(self):
    # Initialize the results storage array
        self.initialize_voltage()
        self.cell.t_results = np.linspace(0, self.tmax, int(self.tmax/self.dt)+1)

        i = 0
        self.run = True
        while self.run:
            self.update()
            if i % 10000 == 0:
                print('Time: {}'.format(int(self.time)))
            i += 1
        self.tcount = 0






