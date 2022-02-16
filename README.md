# PGDrome
A FEniCS based python module of the Proper Generalized Decomposition (PGD) method. 

## Requirements
You need a working installation of FEniCS (including DOLFIN python interface). 

## Installation
Using setup.py
```
git clone https://github.com/BAMresearch/PGDrome.git
cd relpgd
python3 -m pip install -e .
```
or
```
python setup.py install
```
-e flag creates a symlink instead of copying it. So modifying does not require another installation.

uninstall with `python3 -m pip uninstall relpgd` 

test with pytest or standalone
```
pytest tests
```
