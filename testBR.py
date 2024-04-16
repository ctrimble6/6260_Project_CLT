import beelerReuter as br
import numpy as np
import matplotlib.pyplot as plt

# Create an instance of the Beeler-Reuter model
model = br.BeelerReuter()

# Initial Conditions are Hardcoded for now

# Well, might as well try to run the model
voltage = model.run_simulation(0.001, 1, 40, method = 'forward_euler')

# Plot the results
nT = len(voltage)
tpoints = np.linspace(0, 1, nT)
plt.plot(tpoints, voltage)
plt.show()

