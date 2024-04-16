# 6260 Project: Bidomain Solver for Cardiac Dynamics
Author: Casey Lee-Trimble

## Overview
This repository contains the Bidomain Solver, a Python module under development being designed to solve the bidomain equations in cardiac dynamics. The solver employs the Crank-Nicolson method for numerical integration and is structured to be flexible and expandable. It supports simulations in 1D initially, with the architecture in place to extend to 2D and 3D. 

- Main Bidomain Solver evolves the transmembrane potential and the internal potential of the cell
- Beeler Reuter Model solves for the transmembrane ion currents accoring to Beeler Reuter (1977)

## Progress
The Beeler Reuter module successfully creates a lookup table for the rate constants that solve for the ion channel gate parameters. The simulation seems to run fine, but poking it with a stimulus current does not have the intended effect.

## Bugs
The Beeler Reuter model runs, but does not give a physical result yet. Going to investigate the initialization and the handling of the gate update.

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

##
