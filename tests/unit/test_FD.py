"""
    check FD formulation working within fenics

    problem rho cp \partial{T}{dt} = q(t)
    solve for T with backward euler as loop with FD matrix and with FEM
"""

import unittest
import dolfin
import fenics
import numpy as np
import scipy as sp

from pgdrome.solver import FD_matrices


class ref_solution:
    def __init__(self, Vs=None, param=None, meshes=None, q=None):

        self.Vs = Vs  # Location
        self.param = param  # Parameters
        self.meshes = meshes  # Meshes
        self.q = q  # heat source expression

    def run(self):
        T = dolfin.Function(self.Vs)

        T_tmp = np.zeros(len(T.vector()[:]))
        T_tmp[0] = self.param["T_amb"]

        idx_sort = np.argsort(self.Vs.tabulate_dof_coordinates()[:].flatten())
        time_points = np.sort(self.Vs.tabulate_dof_coordinates()[:].flatten())

        for i in range(1, len(time_points)):
            T_tmp[i] = T_tmp[i - 1] + (time_points[i] - time_points[i - 1]) / (
                self.param["rho"] * self.param["c_p"]
            ) * self.q(time_points[i])

        T.vector()[:] = T_tmp[idx_sort]

        return T


class FD_solution:
    def __init__(self, Vs=None, param=None, meshes=None, q=None):
        self.Vs = Vs  # Location
        self.param = param  # Parameters
        self.meshes = meshes  # Meshes
        self.q = q  # heat source expression

    def run(self):
        T = dolfin.Function(self.Vs)

        x_dofs = np.array(self.Vs.tabulate_dof_coordinates()[:].flatten())
        idx_sort = np.argsort(x_dofs)
        Mt, _, D1_upt = FD_matrices(self.Vs.tabulate_dof_coordinates()[idx_sort])

        # store re_sorted according dofs!
        M1 = Mt[idx_sort, :][:, idx_sort]

        # resort D1
        D1_up = D1_upt[idx_sort, :][:, idx_sort]

        # interpolate right hand side
        Q = dolfin.interpolate(self.q, self.Vs).vector()[:]

        # set up initial condition
        IC = np.zeros(len(Q))
        IC[-1] = self.param["T_amb"]

        # set up shorted heat equation
        Amat = self.param["rho"] * self.param["c_p"] * D1_up
        Fvec = M1 @ Q - self.param["rho"] * self.param["c_p"] * D1_up @ IC

        # set matrices for initial condition
        Fvec[-1] = 0
        Amat[:, -1] = 0
        Amat[-1, :] = 0
        Amat[-1, -1] = 1

        # solve problem
        vec_tmp = sp.sparse.linalg.spsolve(Amat.tocsr(), Fvec)

        # add inital condition
        T.vector()[:] = vec_tmp + IC

        return T


class FEM_solution:
    def __init__(self, Vs=None, param=None, meshes=None, q=None):

        self.Vs = Vs  # Location
        # self.Vs = dolfin.FunctionSpace(Vs.mesh(), 'CG', 3) # to improve solution
        self.param = param  # Parameters
        self.meshes = meshes  # Meshes
        self.q = q  # heat source expression

    def run(self):

        # boundary condition
        def init(x, on_boundary):
            return x < 0.0 + 1e-5

        bc = dolfin.DirichletBC(self.Vs, self.param["T_amb"], init)

        # set up functions
        T = fenics.TrialFunction(self.Vs)
        v = fenics.TestFunction(self.Vs)

        # set up problem
        a = self.param["rho"] * self.param["c_p"] * T.dx(0) * v * dolfin.dx
        l = self.q * v * dolfin.dx

        # solve problem
        T = fenics.Function(self.Vs)
        fenics.solve(a == l, T, bcs=bc)

        # # add inital condition
        # T.vector()[-1] = self.param["T_amb"]

        return T


class PGDproblem(unittest.TestCase):
    def setUp(self):
        # global parameters
        self.ord = 1  # has to be one! because of defined mapping from FD matrix Euler!

        self.ranges = [0.0, 50]  # time intervall
        self.elem = 200

        self.param = {
            "rho": 71,
            "c_p": 31,
            "k": 100,
            "P": 250,
            "T_amb": 25,
        }  # material density [kg/m³]  # heat conductivity [W/m°C] # specific heat capacity [J/kg°C]

        # self.plotting = True
        self.plotting = False

    def TearDown(self):
        pass

    def test_solver(self):
        mesh_t = dolfin.IntervalMesh(self.elem, self.ranges[0], self.ranges[1])
        Vs_t = dolfin.FunctionSpace(mesh_t, "CG", self.ord)
        q = dolfin.Expression(
            "x[0] < 5 ? 0 : (x[0] > 20 ? 0 : Q)", degree=1, Q=self.param["P"]
        )

        # reference backward euler
        Tref = ref_solution(Vs=Vs_t, meshes=mesh_t, param=self.param, q=q).run()
        # print(Tref.compute_vertex_values()[:])

        TFD = FD_solution(Vs=Vs_t, meshes=mesh_t, param=self.param, q=q).run()
        # print(TFD.compute_vertex_values()[:])

        TFEM = FEM_solution(Vs=Vs_t, meshes=mesh_t, param=self.param, q=q).run()
        # print(TFEM.compute_vertex_values()[:])

        # check errors
        # FD == reference
        error1 = dolfin.errornorm(TFD, Tref)
        error2 = dolfin.errornorm(TFEM, Tref)
        print("error FD - ref", error1, "error FEM - ref", error2)
        self.assertTrue(error1 < 1e-8)
        self.assertTrue(error2 > error1)  # FEM discretization not useful here!

        if self.plotting:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(
                mesh_t.coordinates()[:],
                Tref.compute_vertex_values()[:],
                "-*r",
                label="ref",
            )
            plt.plot(
                mesh_t.coordinates()[:],
                TFD.compute_vertex_values()[:],
                "-*b",
                label="FD",
            )
            plt.plot(
                mesh_t.coordinates()[:],
                TFEM.compute_vertex_values()[:],
                "-*g",
                label="FEM",
            )
            plt.legend()
            plt.show()


if __name__ == "__main__":
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging

    logging.basicConfig(level=logging.INFO)

    unittest.main()
