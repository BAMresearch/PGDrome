# PGDrome
[![Tests for pgdrome](https://github.com/BAMresearch/PGDrome/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/BAMresearch/PGDrome/actions/workflows/tests.yml)

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
 
# Conda environment with dolfin

* create conda environment with requirements from file
  * globally in user miniconda folder
    ```
    cd PGDrome
    conda env create -f environment.yml
    conda activate pgdrome
    ```
  * within main project folder
    ```
    cd PGDrome
    conda env create --prefix ./conda-env -f environment.yml
    (conda env update --prefix ./conda-env -f environment.yml --prune)
    conda activate ./conda-env
    ```

# Install module pgdrome in other projects
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

# note
* fenicstools (required for special sensor evaluations install only if required!!!)
```
git clone https://github.com/mikaem/fenicstools.git
cd fenicstools
python3 setup.py install
python3 -m pip install cppimport
```



