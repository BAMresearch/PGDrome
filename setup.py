from setuptools import setup, find_packages

extensions = []

setup(name='pgdrome',
      version='0.1',
      description='A FEniCS based python module of the Proper Generalized Decomposition (PGD) method.',
      url='https://github.com/BAMresearch/PGDrome',
      author='Annika Robens-Radermacher, Dominic Strobl, Aratz Garcia Llona',
      author_email='annika.robens-radermacher@bam.de',
      license='MIT',
      packages=find_packages(),
      package_data={'': ['README.md']},
      install_requires=['numpy',
                        'scipy',
                        'lxml',
                        'h5py',
                        ],
      zip_safe=True,
      test_suite="pytest",
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-cov', 'pytest-pylint'],
      ext_modules=extensions, )
