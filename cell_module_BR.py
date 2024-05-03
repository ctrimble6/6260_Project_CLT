import numpy as np
import matplotlib.pyplot as plt
from math import *

class Cell_BR():
    """Beeler-Reuter Model for Cardiac Cell Dynamics
    Inputs:
    initial_V (float): Initial voltage of the cell
    dt (float): Timestep size
    num_cells (int): Number of cells in the tissue
    
    Outputs:
    I_total (np.ndarray): Total current output of the cell model
    """
    
    def __init__(self,initial_V,dt=0.01,num_cells=1) -> None:

        # Beeler-Reuter Parameters
        # These are the constants used in the generalized
        # rate constant equation in Beeler Reuter (1977)
        print("Initializing Beeler-Reuter Model...")
        self.C = np.zeros([12,7])
        self.C[0] = [0.0005,0.083,50,0,0,0.057,1] # ax1
        self.C[1] = [0.0013,-0.06,20,0,0,-0.04,1] # bx1
        self.C[2] = [0,0,47,-1,47,-.1,-1]         # am
        self.C[3] = [40,-.056,72,0,0,0,0]         # bm
        self.C[4] = [0.126,-.25,77,0,0,0,0]       # ah
        self.C[5] = [1.7,0,22.5,0,0,-.082,1]      # bh
        self.C[6] = [0.055,-.25,78,0,0,-.2,1]     # aj
        self.C[7] = [0.3,0,32,0,0,-.1,1]          # bj
        self.C[8] = [0.095,-.01,-5,0,0,-.072,1]   # ad
        self.C[9] = [0.07,-.017,44,0,0,.05,1]     # bd
        self.C[10] = [0.012,-0.008,28,0,0,.15,1]  # af
        self.C[11] = [0.0065,-0.02,30,0,0,-.2,1]  # bf

        # Capacitance and Conductances
        self.g_Na = 4.0    # Maximum conductance of fast inward current
        self.g_NaC = 0.003 # Maximum conductance of Na-Ca exchange current
        self.g_s = 0.09    # Maximum conductance of slow inward current
        self.g_x1 = 0.8    # Maximum conductance of time-activated outward current
        self.g_k = 0.35    # Maximum conductance of Potassium outward current
        self.Cminv = 1     # Membrane capacitance

        # Initial Conditions
        self.V = initial_V
        self.Ca = 10**(-7)*np.ones(num_cells)

        # Activation Potentials
        self.E_Na = 50
        self.E_s = -82.3 - 13.0287*np.log(self.Ca) # Nernst potential for slow inward current
        self.E_x = -77

        # Simulation Parameters
        self.dt = dt

        # Lookup Table Parameters
        self.dV = 0.01 # This step size was chosen, and determines the resolution of the lookup tables
        self.bounds = [-100,100] # In mV, the AP is not expected to go beyond these bounds
        self.Vs = np.arange(self.bounds[0],self.bounds[1],self.dV)
        self.ab_table = np.zeros((12,len(self.Vs)))
        self.steady_state_table = np.zeros((6,len(self.Vs)))
        self.time_constant_table = np.zeros((6,len(self.Vs)))
        self.lookup = np.zeros((6,len(self.Vs)))
        self.I_total = np.zeros(num_cells)


        # Initialize the lookup tables
        for i in range(12):
            for j in range(len(self.Vs)): 
                self.ab_table[i,j] = self.get_rate_constants(self.Vs[j],i) # Alpha and Beta terms

        z = 0
        for i in range(6):
            for j in range(len(self.Vs)):
                self.steady_state_table[i,j] = self.get_steady_state(z,j) # Steady state values
                self.time_constant_table[i,j] = self.get_time_constant(z,j) # Time constants
            z += 2 

        # Initialize the gates at their steady state values
        v_index = np.asarray((self.V - self.bounds[0])/self.dV, dtype=int)


        self.x1 = np.zeros(num_cells)
        self.m = np.zeros(num_cells)
        self.h = np.zeros(num_cells)
        self.j = np.zeros(num_cells)
        self.d = np.zeros(num_cells)
        self.f = np.zeros(num_cells)

        self.x1[:] = self.steady_state_table[0,v_index]
        self.m[:] = self.steady_state_table[1,v_index]
        self.h[:] = self.steady_state_table[2,v_index]
        self.j[:] = self.steady_state_table[3,v_index]
        self.d[:] = self.steady_state_table[4,v_index]
        self.f[:] = self.steady_state_table[5,v_index]

        print("Initialization complete")

       

    ## Functions for initializing the rate constants and lookup tables ##    
    def get_rate_constants(self,V,i) -> float:
        # Calculate rate constants
        numerator = self.C[i,0]*np.exp(self.C[i,1]*(V + self.C[i,2])) + self.C[i,3]*(V + self.C[i,4])
        denominator = np.exp(self.C[i,5]*(V + self.C[i,2])) + self.C[i,6]
        rc = np.divide(numerator,denominator)
        return rc
            
    def get_steady_state(self,z,j) -> float:
        return self.ab_table[z,j]/(self.ab_table[z,j] + self.ab_table[z+1,j])
    
    def get_time_constant(self,z,j) -> float:
        return 1/(self.ab_table[z,j] + self.ab_table[z+1,j])
    
    ## Functions for updating the gates ##

    def get_v_index(self) -> np.ndarray:
        return np.asarray((self.V - self.bounds[0])/self.dV,dtype=int)

    def update_x1(self,v_index) -> None:
        #self.x1 = self.steady_state_table[0,v_index] + (self.x1 - self.steady_state_table[0,v_index])*np.exp(-np.divide(self.dt,self.time_constant_table[0,v_index]))
        self.x1 = self.x1 + self.dt*(self.steady_state_table[0,v_index] - self.x1)/self.time_constant_table[0,v_index] 
    def update_m(self,v_index) -> None:
        #self.m = self.steady_state_table[1,v_index] + (self.m - self.steady_state_table[1,v_index])*np.exp(-self.dt/self.time_constant_table[1,v_index])
        self.m = self.m + self.dt*(self.steady_state_table[1,v_index] - self.m)/self.time_constant_table[1,v_index]
    def update_h(self,v_index) -> None:
        #self.h = self.steady_state_table[2,v_index] + (self.h - self.steady_state_table[2,v_index])*np.exp(-self.dt/self.time_constant_table[2,v_index])
        self.h = self.h + self.dt*(self.steady_state_table[2,v_index] - self.h)/self.time_constant_table[2,v_index]
    def update_j(self,v_index) -> None:
        #self.j = self.steady_state_table[3,v_index] + (self.j - self.steady_state_table[3,v_index])*np.exp(-self.dt/self.time_constant_table[3,v_index])
        self.j = self.j + self.dt*(self.steady_state_table[3,v_index] - self.j)/self.time_constant_table[3,v_index]
    def update_d(self,v_index) -> None:
        #self.d = self.steady_state_table[4,v_index] + (self.d - self.steady_state_table[4,v_index])*np.exp(-self.dt/self.time_constant_table[4,v_index])
        self.d = self.d + self.dt*(self.steady_state_table[4,v_index] - self.d)/self.time_constant_table[4,v_index]
    def update_f(self,v_index) -> None:
        #self.f = self.steady_state_table[5,v_index] + (self.f - self.steady_state_table[5,v_index])*np.exp(-self.dt/self.time_constant_table[5,v_index])
        self.f = self.f + self.dt*(self.steady_state_table[5,v_index] - self.f)/self.time_constant_table[5,v_index]
    
    def update_gates(self) -> None:
        v_index = self.get_v_index()
        self.update_x1(v_index)
        self.update_m(v_index)
        self.update_h(v_index)
        self.update_j(v_index)
        self.update_d(v_index)
        self.update_f(v_index)

    ## Functions for calculating the currents ##
    def I_x1(self) -> np.ndarray:
        current = np.asarray(self.g_x1*self.x1*(np.exp(0.04*(self.V + 77)) - 1)/(np.exp(0.04*(self.V + 35))),dtype=np.float64)
        return current
    def I_Na(self) -> np.ndarray:
        current = np.asarray((self.g_Na*self.m*self.m*self.m*self.h*self.j+self.g_NaC)*(self.V - self.E_Na),dtype=np.float64)
        return current
    def I_s(self) -> np.ndarray:
        E_s = -82.3 - 13.0287*np.log(self.Ca) # Nernst potential for slow inward current
        current = np.asarray(self.g_s*self.d*self.f*(self.V - E_s),dtype=np.float64)
        return current
    
    def I_k(self) -> np.ndarray:
        A = 4 * (np.exp(0.04 * (self.V + 85)) - 1)
        B = np.exp(0.08 * (self.V + 53)) + np.exp(0.04 * (self.V + 53))
        C = np.where(self.V != 23, 
                     0.2 * (self.V + 23) / (1 - np.exp(-0.04 * (self.V + 23))),
                     5)
        current = np.asarray(self.g_k * (A / B + C),dtype=np.float64)
        return current
    
    ## Updating the Calcium Concentration ##

    def update_Ca(self) -> None:
        self.Ca = self.Ca + self.dt*(-self.I_s()*10**(-7) + 0.07*(10**(-7) - self.Ca))

    ## Update the Cell ##
    def update(self,V_new) -> None:
        # I_total is the main output of this model.
        self.V = V_new # New voltage being passed in from Bidomain Solver
        self.update_gates() # Update the gates using new voltage
        self.update_Ca() # Update the Calcium Concentration based on new I_s
        self.I_total[:] = np.asarray(self.I_x1() + self.I_Na() + self.I_s() + self.I_k()) # Calculate the total current 

    def forward_euler_test(self) -> None:
        t = 0
        v_results_cell = []
        m_results = []
        h_results = []
        j_results = []
        d_results = []
        f_results = []
        Ca_results = []
        t_results = []
        
        while t < 1000:
            
            self.update(self.V)
            self.V = self.V - self.dt*(self.I_total*self.Cminv)
            
            # A stimulus
            if t < 100.5:
                if t > 100:
                    self.V = self.V + 1

            # A second stimulus
            #if t < 400.5:
            #    if t > 400:
            #        self.V = self.V + 1
            v_results_cell.append(self.V)
            m_results.append(self.m)
            h_results.append(self.h)
            j_results.append(self.j)
            d_results.append(self.d)
            f_results.append(self.f)
            Ca_results.append(self.Ca)
            t_results.append(t)
            t += self.dt
        
        m_results = np.asarray(m_results)
        h_results = np.asarray(h_results)
        j_results = np.asarray(j_results)
        d_results = np.asarray(d_results)
        f_results = np.asarray(f_results)
        Ca_results = np.asarray(Ca_results)
        t_results = np.asarray(t_results)

        V_norm = self.normalize_AP(v_results_cell)

        plt.figure()
        #plt.plot(t_results,m_results,label='m')
        #plt.plot(t_results,h_results,label='h')
        #plt.plot(t_results,j_results,label='j')
        plt.plot(t_results,V_norm,label='$T_1=100$msec')
        plt.ylabel('Voltage (Normalized)')
        plt.xlabel('Time (ms)')
        plt.legend()
        plt.show()

        APD90 = self.measure_APD90(V_norm)
        print("APD90: ",APD90,"ms")

    def measure_APD90(self,V_norm) -> float:
        # Right now, this only works for one AP, not a train
        V_max = np.max(V_norm)
        V_min = np.min(V_norm)
        V_90 = V_min + 0.1*(V_max - V_min)
        indices = np.where(V_norm > V_90) # Find where V_norm > V_90
        start = indices[0][0] # Select the first index where V_norm > V_90
        end = indices[0][-1] # Select the last index where V_norm > V_90
        APD90 = (end - start)*self.dt
        return APD90
    
    def normalize_AP(self,V) -> float:
        V_max = np.max(V)
        V_min = np.min(V)
        return (V - V_min)/(V_max - V_min)
    
    def count_peaks(self,V) -> int:
        # In anticipation of a train of APs
        peaks = 0
        for i in range(1,len(V)-1):
            if V[i] > V[i-1] and V[i] > V[i+1]:
                peaks += 1
        return peaks



        



