'''
    test for PGD Class functions and generation

    using arbitrary DOLFIN functions and meshes as PGD modes/ attributes

    u(x,p,E) = x*x * p * 1/E

'''

import unittest
import os
import tempfile
import shutil

import dolfin

from pgdrome.model import PGD


class TestPGD(unittest.TestCase):
    def setUp(self):
        # define some meshes with dolfin
        mesh_x = dolfin.IntervalMesh(50, 0., 1.0)
        V_x = dolfin.FunctionSpace(mesh_x, 'CG', 1)

        mesh_p = dolfin.IntervalMesh(10, 0., 2.0)
        V_p = dolfin.FunctionSpace(mesh_p, 'CG', 1)

        mesh_E = dolfin.IntervalMesh(10, 0.5, 1.0)
        V_E = dolfin.FunctionSpace(mesh_E, 'CG', 2)

        # define some PGD_modes as dolfin fcts
        PGD_func = [list(), list(), list()]
        for nn in range(2):
            PGD_func[0].append(dolfin.project(dolfin.Expression('x[0]*x[0]', degree=10), V_x))
            PGD_func[1].append(dolfin.project(dolfin.Expression('x[0]', degree=10), V_p))
            PGD_func[2].append(dolfin.project(dolfin.Expression('1.0/x[0]', degree=10), V_E))

        #create a pgd class from that
        self.pgd_test = PGD(name='Test', n_modes=1,
                    fmeshes=[mesh_x, mesh_p, mesh_E],
                    pgd_modes=PGD_func,
                    name_coord=['X', 'P', 'E'],
                    modes_info=['U_x', 'Node', 'Scalar'], verbose=True)

        # self.pgd_test.print_info()

        # save model as pxdmf file in a temp file directory
        self.filepathPGD = tempfile.mkdtemp() #os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests'))
        self.pgd_test.write_pxdmf(self.filepathPGD, False)
        # write hdf5 data files
        self.pgd_test.write_hdf5(self.filepathPGD)

        # define PGD coordinates for tests
        self.E = 0.75
        self.P = 0.75


    def tearDown(self):
        # clear up temporary basedir folder
        print("DELETE", self.filepathPGD)
        shutil.rmtree(self.filepathPGD)


    def test_load_and_evaluate(self):
        # load model from pxdmf
        pgd_load = PGD().load_pxdmf(os.path.join(self.filepathPGD, 'Test.pxdmf'), True)

        # define interpolation fct with dolfin function spaces
        pgd_load.mesh[0].attributes[0].interpolationInfo = {'name': 1, 'family': 'CG', 'degree': 1, '_type': 'scalar'}
        pgd_load.mesh[1].attributes[0].interpolationInfo = {'name': 1, 'family': 'CG', 'degree': 1, '_type': 'scalar'}
        pgd_load.mesh[2].attributes[0].interpolationInfo = {'name': 1, 'family': 'CG', 'degree': 2, '_type': 'scalar'}
        pgd_load.mesh[1].attributes[0].print_info()

        pgd_load.create_interpolation_fcts([0, 1, 2], 0)

        # check interpolation functions
        self.assertAlmostEqual(0.8**2, pgd_load.mesh[0].attributes[0].interpolationfct[0](0.8), places=3)
        self.assertAlmostEqual(0.8, pgd_load.mesh[1].attributes[0].interpolationfct[0](0.8), places=3)
        self.assertAlmostEqual(1/0.8, pgd_load.mesh[2].attributes[0].interpolationfct[0](0.8), places=3)
        # self.assertAlmostEqual()

        # check evaluation
        evaluate = pgd_load.evaluate(0, [1, 2], [self.P, self.E], 0)
        x_set = 0.5
        # print(f'PGD value for E={self.E}, p={self.P} and x={x_set}: {evaluate(x_set)}')
        # print(f'Value exact: {x_set**2 * self.P * 1/self.E}')
        self.assertAlmostEqual(x_set**2 * self.P * 1/self.E, evaluate(x_set), places=1)


if __name__ == "__main__":
    unittest.main()
