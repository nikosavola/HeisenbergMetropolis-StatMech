# Heisenberg model in two dimensions

[![CI](https://img.shields.io/github/workflow/status/nikosavola/HeisenbergMetropolis-StatMech/CI)](https://github.com/nikosavola/HeisenbergMetropolis-StatMech/actions/workflows/ci.yaml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nikosavola/HeisenbergMetropolis-StatMech/HEAD?labpath=analysis.ipynb)


This repository contains source code for a computational project on the classical Heisenberg model in two dimensions using a standard Monte Carlo Metropolis algorithm for the Ising model.

Project is done for the Aalto University course [PHYS-E0415 Statistical Mechanics](https://courses.aalto.fi/courses/s/course/a053X000012QxjCQAS/statistical-mechanics-d?language=en_US).

## Installation

Using Jupyter Notebooks is preferred. Necessary Python packages can be installed with:
```bash
pip install -r requirements.txt
```

## Usage

[`heisenberg_2d.py`](heisenberg_2d.py) contains the implementation, which can be run using the [`analysis.ipynb`](analysis.ipynb) notebook.

Simulations can be run independently with e.g.
```bash
python3 run_heisenberg.py --N 10 --H 0 0.25 0.75 --steps 4000 --temp 0.3 10.5 500
```
The parameters can in turn be effortlessly swept with something like the Bash-script given as an example [`run_simulations.sh`](run_simulations.sh):
```bash
chmod a+x run_simulations.sh
./run_simulations.sh
```

Link to (private) Overleaf [here](https://www.overleaf.com/project/6176a385e66750335f81c27d).
