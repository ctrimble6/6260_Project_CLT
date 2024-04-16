"""Beeler Reuter Model"""
import numpy as np
import matplotlib.pyplot as plt

# import pandas as pd

# C[0] - msec^-1
# C[1] - mV^-1
# C[2] - mV
# C[3] - (mV*msec)^-1
# C[4] - mV
# C[5] - mV^-1
# C[6] - dimensionless (1, 0, or -1)

class BeelerReuter:
    def __init__(self):
        # Define the parameters of the Beeler Reuter Model from a CSV file
        # self.parameters = self.load_parameters(parameter_file) for later

        # Initialize the model -- For now, I am hardcoding the parameters. Will use Pandas later
        self.Cax1 = [0.0005,0.083,50,0,0,0.057,1]
        self.Cbx1 = [0.0013,-0.06,20,0,0,-0.04,1] 
        self.Cam = [0,0,47,-1,47,-.1,-1]
        self.Cbm = [40,-.056,72,0,0,0,0]
        self.Cah = [0.126,-.25,77,0,0,0,0]
        self.Cbh = [1.7,0,22.5,0,0,-.082,1]
        self.Caj = [0.055,-.25,78,0,0,-.2,1]
        self.Cbj = [0.3,0,32,0,0,-.1,1]
        self.Cad = [0.095,-.01,-5,0,0,-.072,1]
        self.Cbd = [0.07,-.017,44,0,0,.05,1]
        self.Caf = [0.012,-0.008,28,0,0,.15,1]
        self.Cbf = [0.0065,-0.02,30,0,0,-.2,1]

        self.g_Na = 4.0 # Maximum conductance of fast inward current
        self.g_NaC = 0.003 # Maximum conductance of Na-Ca exchange current
        self.g_s = 0.09 # Maximum conductance of slow inward current
        self.g_x = 0.3 # Maximum conductance of time-activated outward current
        self.calcium = 2*10**(-7) # Initial calcium concentration
        self.E_Na = 50
        self.E_s = -82.3 - 13.0287*np.log(self.calcium) # Nernst potential for slow inward current
        self.E_x = -77
        self.Cm = 1.0 # Membrane capacitance

        self.v_tm = -84.622 # Initialize membrane potential  
        self.init_rcs()
        self.create_lookup_table()
        self.steady_state()
        self.time_constant()
        self.init_gates()

        self.I_Na = self.getI_Na() # Fast inward current
        self.I_s = self.getI_s() # Slow inward current
        self.I_x = self.getI_x() # Time-activated outward current
        self.I_K = self.getI_K() # Time-independent potassium current


    # Generalized Rate Function
    def getrc(self, vm, C):
    # Handles division by zero or overflow errors gracefully
        try:
            numerator = C[0]*np.exp(C[1]*(vm + C[2])) + C[3]*(vm + C[4])
            denominator = np.exp(C[5]*(vm + C[2])) + C[6]
            rc = numerator / denominator if denominator != 0 else 0
        except OverflowError:
            rc = float('inf')  # or other appropriate large number
        return rc

    def init_rcs(self):
        # Update gating variables and other state variables
        vm = self.v_tm
        self.ax1 = self.getrc(vm,self.Cax1) # Time-Activated outward current
        self.bx1 = self.getrc(vm,self.Cbx1)
        
        self.am = self.getrc(vm,self.Cam) # Activation gate for fast inward current
        self.bm = self.getrc(vm,self.Cbm)
        
        self.ah = self.getrc(vm,self.Cah) # Fast inactivation gate for fast inward current
        self.bh = self.getrc(vm,self.Cbh)
        
        self.aj = self.getrc(vm,self.Caj) # Slow inactivation gate for fast inward current
        self.bj = self.getrc(vm,self.Cbj)
        
        self.ad = self.getrc(vm,self.Cad) # Activation gate for slow inward current
        self.bd = self.getrc(vm,self.Cbd)
        
        self.af = self.getrc(vm,self.Caf) # Inactivation gate for slow inward current
        self.bf = self.getrc(vm,self.Cbf)

    # Create Lookup Table for the Rate Constants
    def create_lookup_table(self):
        # Create a lookup table for the rate constants
        # Initialize the lookup tables
        self.ax1tab = np.zeros(200)
        self.bx1tab = np.zeros(200)
        self.amtab = np.zeros(200)
        self.bmtab = np.zeros(200)
        self.ahtab = np.zeros(200)
        self.bhtab = np.zeros(200)
        self.ajtab = np.zeros(200)
        self.bjtab = np.zeros(200)
        self.adtab = np.zeros(200)
        self.bdtab = np.zeros(200)
        self.aftab = np.zeros(200)
        self.bftab = np.zeros(200)   
        
        for i in range(-100, 100):
            self.ax1tab[i] = self.getrc(i, self.Cax1)
            self.bx1tab[i] = self.getrc(i, self.Cbx1)
            self.amtab[i] = self.getrc(i, self.Cam)
            self.bmtab[i] = self.getrc(i, self.Cbm)
            self.ahtab[i] = self.getrc(i, self.Cah)
            self.bhtab[i] = self.getrc(i, self.Cbh)
            self.ajtab[i] = self.getrc(i, self.Caj)
            self.bjtab[i] = self.getrc(i, self.Cbj)
            self.adtab[i] = self.getrc(i, self.Cad)
            self.bdtab[i] = self.getrc(i, self.Cbd)
            self.aftab[i] = self.getrc(i, self.Caf)
            self.bftab[i] = self.getrc(i, self.Cbf)

    # Interpolate the Rate Constants from the Lookup Table
    def linterp_rctab(self):
        # Use linear interpolation to get the rate constants
        vm = self.v_tm
        vmlow = int(vm)
        vmhigh = vmlow + 1
        self.ax1 = (self.ax1tab[vmlow]*(vmhigh - vm) + self.ax1tab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.bx1 = (self.bx1tab[vmlow]*(vmhigh - vm) + self.bx1tab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.am = (self.amtab[vmlow]*(vmhigh - vm) + self.amtab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.bm = (self.bmtab[vmlow]*(vmhigh - vm) + self.bmtab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.ah = (self.ahtab[vmlow]*(vmhigh - vm) + self.ahtab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.bh = (self.bhtab[vmlow]*(vmhigh - vm) + self.bhtab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.aj = (self.ajtab[vmlow]*(vmhigh - vm) + self.ajtab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.bj = (self.bjtab[vmlow]*(vmhigh - vm) + self.bjtab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.ad = (self.adtab[vmlow]*(vmhigh - vm) + self.adtab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.bd = (self.bdtab[vmlow]*(vmhigh - vm) + self.bdtab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.af = (self.aftab[vmlow]*(vmhigh - vm) + self.aftab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)
        self.bf = (self.bftab[vmlow]*(vmhigh - vm) + self.bftab[vmhigh]*(vm - vmlow))/(vmhigh - vmlow)

    def steady_state(self):
        self.x1_inf = self.ax1/(self.ax1 + self.bx1)
        self.m_inf = self.am/(self.am + self.bm)
        self.h_inf = self.ah/(self.ah + self.bh)
        self.j_inf = self.aj/(self.aj + self.bj)
        self.d_inf = self.ad/(self.ad + self.bd)
        self.f_inf = self.af/(self.af + self.bf)

    def time_constant(self):
        self.tau_x1 = 1/(self.ax1 + self.bx1)
        self.tau_m = 1/(self.am + self.bm)
        self.tau_h = 1/(self.ah + self.bh)
        self.tau_j = 1/(self.aj + self.bj)
        self.tau_d = 1/(self.ad + self.bd)
        self.tau_f = 1/(self.af + self.bf)

    def init_gates(self):
        self.x1 = self.x1_inf
        self.m = self.m_inf
        self.h = self.h_inf
        self.j = self.j_inf
        self.d = self.d_inf
        self.f = self.f_inf

    # Update the gating variables
    def update_gating_variables(self, dt):
        self.linterp_rctab()
        self.steady_state()
        self.time_constant()
        self.x1 += dt*(self.x1_inf - self.x1)/self.tau_x1
        self.m += dt*(self.m_inf - self.m)/self.tau_m
        self.h += dt*(self.h_inf - self.h)/self.tau_h
        self.j += dt*(self.j_inf - self.j)/self.tau_j
        self.d += dt*(self.d_inf - self.d)/self.tau_d
        self.f += dt*(self.f_inf - self.f)/self.tau_f
        self.calcium += (-10**(-7)*self.I_s + 0.07*(10**(-7) - self.calcium))*dt

    # Fast Inward Current (I_Na)
    def getI_Na(self):
        return (self.g_Na*(self.m**(3))*self.h*self.j + self.g_NaC)*(self.v_tm - self.E_Na)

    # Slow Inward Current (I_s)
    def getI_s(self):
        self.E_s = -82.3 - 13.0287*np.log(self.calcium)
        return self.g_s*self.d*self.f*(self.v_tm - self.E_s)

    # Time-Activated Outward Current (I_x)
    def getI_x(self):
        return 0.8*self.x1*(np.exp(0.04*(self.v_tm + 77)) - 1)/(np.exp(0.04*(self.v_tm + 35)))

    # Time-Independent Potassium Current (I_K)
    def getI_K(self):
        return 0.35 * ( 4*(np.exp( 0.04 * (self.v_tm + 85) ) - 1)/(np.exp( 0.08 * (self.v_tm + 53) ) 
                                                                + np.exp( 0.04 * (self.v_tm + 53))) 
                                                                - 0.2*(self.v_tm + 23)/(np.exp(-0.04*(self.v_tm + 23))))
    
    def calculate_currents(self, voltage):
        self.v_tm = voltage
        self.update_gating_variables(dt=0.001)
        self.I_Na = self.getI_Na()
        self.I_s = self.getI_s()
        self.I_x = self.getI_x()
        self.I_K = self.getI_K()
        return self.I_Na + self.I_s + self.I_x + self.I_K
    
    def forward_euler(self, dt, I_stim):
        # Forward Euler Method
        self.v_tm += dt*(I_stim - self.calculate_currents(self.v_tm))/self.Cm
        self.update_gating_variables(dt)
        return self.v_tm
    
    def runge_kutta4(self, dt, I_stim):
        # Runge-Kutta 4 Method
        v1 = self.v_tm
        I1 = self.calculate_currents(v1)
        v2 = v1 + dt*I1/(2*self.Cm)
        I2 = self.calculate_currents(v2)
        v3 = v1 + dt*I2/(2*self.Cm)
        I3 = self.calculate_currents(v3)
        v4 = v1 + dt*I3/self.Cm
        I4 = self.calculate_currents(v4)
        self.v_tm += dt*(-(I1 + 2*I2 + 2*I3 + I4)/6 + I_stim)/self.Cm
        self.update_gating_variables(dt)

    def solve(self, dt, I_stim, method='forward_euler'):
        if method == 'forward_euler':
            return self.forward_euler(dt, I_stim)
        elif method == 'runge_kutta4':
            return self.runge_kutta4(dt, I_stim)
        else:
            raise ValueError('Method not supported')
        
    def run_simulation(self, dt, I_stim, duration, method='forward_euler'):
        # Run the simulation for a specified duration
        num_steps = int(duration/dt)
        voltage = np.zeros(num_steps)
        for i in range(num_steps):
            voltage[i] = self.solve(dt, I_stim, method)
        return voltage
    
    def plot_voltage(self, dt, I_stim, duration, method='forward_euler'):
        # Plot the membrane potential
        voltage = self.run_simulation(dt, I_stim, duration, method)
        time = np.arange(0, duration, dt)
        plt.plot(time, voltage)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.show()



