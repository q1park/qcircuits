# Quantum Circuit Simulator
#### Circuits are represented by sparse matrices and states are represented by sparse vectors

### Notebook for general usage

example_usage.ipynb - Contains examples of the functionality for this simulator

qctools.py - Creates a class for building quantum circuits, evolving states, and outputting human-readable forms

gateset.py - Contains predefined standard gates for building quantum circuits

circuitgrad.py - Contains functions for determining gate rotations via gradient descent

### Notebook for reproducing https://arxiv.org/abs/1903.08257

XYmodel_simulations.ipynb - Contains code for running all simulations and creating all plots that appear in the paper

functions.py - Contains specialized functions for simulating the models described in the paper

