# 6260 Project: Bidomain Solver for Cardiac Dynamics
Author: Casey Lee-Trimble

## Overview
This repository contains the Bidomain Solver, a Python module designed for solving the bidomain equations in cardiac dynamics. The solver employs the Crank-Nicolson method for numerical integration and is structured to be flexible and expandable. It supports simulations in 1D initially, with the architecture in place to extend to 2D and 3D. 

## Features
- **Dimensionality**: Support for 1D simulations with plans for 2D and 3D.
- **Ion Models**: Modular design for implementing different cardiac ion channel models. Comes with a basic model and can be easily extended.
- **Parallelization**: Initially in serial but designed to be parallelized in the future.
- **Configurable Simulation Parameters**: Including time, timestep size, domain shape and size, external pacing functions, and heterogeneous conductivity.

### Requirements
- Python 3.6 or higher
- NumPy
