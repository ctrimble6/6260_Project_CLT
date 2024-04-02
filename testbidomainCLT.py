import bidomainCLT as bidomain
import numpy as np
import matplotlib.pyplot as plt
import time

# Create an instance of the Bidomain class

test_instance = bidomain.BidomainSolver

# Test the function

test = test_instance()
test.initialize_v('step')

# Plot the initial condition
plt.plot(test.v_tm, label='Step Initial Condition', color='black', linestyle='--')
plt.xlim(0,100)
plt.ylim(0,2.5)
plt.xlabel('x')
plt.ylabel('v')

# Test the crank-nicolson initialization
test.cn_init_1d(BC='Dirichlet', het_input='step') # het_input can be 'step', 'random', or 'homogeneous'

# Test the crank-nicolson method
test.cn_solve()

plt.plot(test.v_tm, label= f'Solution at t = {test.time}', color='blue', linestyle='-')
plt.title("Dirichlet BC with Step-wise Heterogeneous Conductivity")
plt.legend()
plt.savefig('solution.png')
plt.show()


