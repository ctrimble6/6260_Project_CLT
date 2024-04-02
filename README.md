# 6260 Project: Bidomain Solver for Cardiac Dynamics
Author: Casey Lee-Trimble

## Overview
This repository contains the Bidomain Solver, a Python module under development being designed to solve the bidomain equations in cardiac dynamics. The solver employs the Crank-Nicolson method for numerical integration and is structured to be flexible and expandable. It supports simulations in 1D initially, with the architecture in place to extend to 2D and 3D. 

## Features
- **Heterogeneous Conductivity**: Limited selection of conductivity profiles. Easy to implement more.
- **Boundary Conditions**: Select BC's as an argument. Two currently supported, more if needed.
### Future Features
- **Dimensionality**: Support for 1D simulations, with plans for 2D and 3D.
- **Ion Models**: Modular design for implementing different cardiac ion channel models. Planning on adding basic model first and building from there.
- **Parallelization**: Initially in serial but trying to write certain modules to be parallelizable in the future.
- **Configurable Simulation Parameters**: Will fine-tune these once the full model is online.

### Requirements
- Python 3.6 or higher
- NumPy
