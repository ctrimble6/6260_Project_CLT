import bidomainCLTdraft1 as bidomain
import numpy as np
import matplotlib.pyplot as plt
import time

# Create an instance of the Bidomain class

test_instance = bidomain.BidomainSolver

# Test the function

test = test_instance()
test.initialize_v('step')

# plot the initial condition and save it to a file
plt.plot(test.v_tm)
plt.xlabel('x')
plt.ylabel('v')
plt.title('Initial Condition')
plt.savefig('initial_condition.png')
plt.show()

# Test the crank-nicolson initialization
test.cn_init_1d(S_input=1.0, BC='Neumann', het_input='homogeneous')

# Test the crank-nicolson method
test.cn_solve()

# plot the solution and save it to a file
plt.plot(test.v_tm)
plt.savefig('solution.png')
plt.xlabel('x')
plt.ylabel('v')
plt.title('Solution at t=1')
plt.show()

