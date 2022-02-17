# PGDrome
A FEniCS based python module of the Proper Generalized Decomposition (PGD) method. 

# Description
* The progressive PGD solver is implemented following publication such as 

> "Recent advances and new challenges in the use of the proper generalized decomposition for solving multidimensional models."
> Chinesta, F., Ammar, A. and Cueto, E. 
> Archive of Computational Methods in Engineering 17/4 (2010): 327-350.

> "A Proper Generalized Decomposition for the solution of elliptic problems in abstract form by using a functional Eckart–Young approach"
> Falco, A. and Nouy, A.
> Journal of Mathematical Analysis and Applications 376/2 (2011): 469-480.

> "A nonintrusive proper generalized decomposition scheme with application in biomechanics"
> Zou, X., Conti, M., Díez,P. and Auricchio, F.
> International Journal of Numerical Methods in Engineering 113/2 (2017): 230-251.
>
> "On the Existence of a Progressive Variational Vademecum based on the Proper Generalized Decomposition for a Class of Elliptic Parameterized Problems"
> Falcó A., Montés, N., Chinesta, F., Hilario, L. and Mora, M.C.
> Journal of Computational and Applied Mathematics 330 (2018): 1093-1107.

# Requirements
* A working installation of FEniCS (including DOLFIN python interface).
* e.g. via conda s. below

# Install 
* Using setup.py
```
git clone https://github.com/BAMresearch/PGDrome.git
cd PGDrome
python3 -m pip install -e .
```
-e flag creates a symlink instead of copying it. So modifying does not require another installation.

uninstall with `python3 -m pip uninstall PGDrome` 

test with pytest or standalone
```
pytest tests
```

# Conda environment with dolfin

* dolfin
```
conda install -c conda-forge fenics mshr 
```
* fenicstools (required for some evaluation fct)
```
git clone https://github.com/mikaem/fenicstools.git
cd fenicstools
python3 setup.py install
python3 -m pip install cppimport
```
* create full conda enviroment with requirements
```
conda create -n <name> -c conda-forge python=3.8 fenics mshr ipython h5py numpy scipy pytest
```
plus fenicstools (see above)

# Unittests & coverage

```
pytest tests --cov=PGDrome tests
coverage html
firefox htmlcov/index.html
```

