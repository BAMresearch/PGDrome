"""
    check FD formulation working within fenics for different boundary conditions

    problem \partial²{T}{dx²} = q(x)
    solve for T with FD matrix and with FEM
"""

import unittest
import dolfin
import fenics
import numpy as np
import scipy as sp

from pgdrome.solver import FD_matrices


class FD_solution_dirichlet:
    def __init__(self, Vs=None, param=None, meshes=None, q=None):
        self.Vs = Vs  # Location
        self.param = param  # Parameters
        self.meshes = meshes  # Meshes
        self.q = q  # heat source expression

    def run(self):
        T = dolfin.Function(self.Vs)

        x_dofs = np.array(self.Vs.tabulate_dof_coordinates()[:].flatten())
        idx_sort = np.argsort(x_dofs)
        _, D2_cent, _ = FD_matrices(self.Vs.tabulate_dof_coordinates()[idx_sort])

        # resort D1
        D2 = D2_cent[idx_sort, :][:, idx_sort]

        # interpolate right hand side
        Q = dolfin.interpolate(self.q, self.Vs).vector()[:]

        # set up initial condition
        IC = np.zeros(len(Q))
        IC[-1] = 100
        IC[0] = self.param["T_amb"]

        # set up shorted heat equation
        Amat = -D2
        Fvec = D2 @ IC

        # set matrices for initial condition
        Fvec[0] = 0
        Fvec[-1] = 0
        Amat[:, 0] = 0
        Amat[0, :] = 0
        Amat[0, 0] = 1
        Amat[:, -1] = 0
        Amat[-1, :] = 0
        Amat[-1, -1] = 1

        # solve problem
        vec_tmp = sp.sparse.linalg.spsolve(Amat.tocsr(), Fvec)

        # add inital condition
        T.vector()[:] = vec_tmp + IC

        return T


class FD_solution_robin:
    def __init__(self, Vs=None, param=None, meshes=None, q=None):
        self.Vs = Vs  # Location
        self.param = param  # Parameters
        self.meshes = meshes  # Meshes
        self.q = q  # heat source expression

    def run(self):
        T = dolfin.Function(self.Vs)

        x_dofs = np.array(self.Vs.tabulate_dof_coordinates()[:].flatten())
        idx_sort = np.argsort(x_dofs)
        Mx, D2_cent, _ = FD_matrices(self.Vs.tabulate_dof_coordinates()[idx_sort])

        # store re_sorted according dofs!
        M = Mx[idx_sort, :][:, idx_sort]

        # resort D1
        D2 = D2_cent[idx_sort, :][:, idx_sort]

        # interpolate right hand side
        Q = dolfin.interpolate(self.q, self.Vs).vector()[:]

        # set up shorted heat equation
        Amat = -D2
        Fvec = M @ Q

        # define space step size
        dx = x_dofs[0] - x_dofs[1]

        # set matrices for initial condition
        Amat[:, 0] = 0
        Amat[0, :] = 0
        Amat[0, 0] = 1 / dx + self.param["h"]
        Amat[0, 1] = -1 / dx
        Amat[1, 0] = -1 / dx
        Amat[:, -1] = 0
        Amat[-1, :] = 0
        Amat[-1, -1] = 1 / dx + self.param["h"]
        Amat[-1, -2] = -1 / dx
        Amat[-2, -1] = -1 / dx

        Fvec[0] = self.param["h"] * self.param["T_amb"]
        Fvec[-1] = self.param["h"] * self.param["T_amb"]

        # solve problem
        vec_tmp = sp.sparse.linalg.spsolve(Amat.tocsr(), Fvec)

        # add inital condition
        T.vector()[:] = vec_tmp

        return T


class FEM_solution_dirichlet:
    def __init__(self, Vs=None, param=None, meshes=None, q=None):
        self.Vs = Vs  # Location
        # self.Vs = dolfin.FunctionSpace(Vs.mesh(), 'CG', 2)  # to improve solution
        self.param = param  # Parameters
        self.meshes = meshes  # Meshes
        self.q = q  # heat source expression

    def run(self):
        # boundary condition
        def left(x, on_boundary):
            return x < np.min(self.Vs.mesh().coordinates()[:]) + 1e-8

        def right(x, on_boundary):
            return x > np.max(self.Vs.mesh().coordinates()[:]) - 1e-8

        bc = [
            dolfin.DirichletBC(self.Vs, 100, left),
            dolfin.DirichletBC(self.Vs, self.param["T_amb"], right),
        ]

        # set up problem
        T = fenics.TrialFunction(self.Vs)
        v = fenics.TestFunction(self.Vs)

        a = fenics.dot(fenics.grad(T), fenics.grad(v)) * fenics.dx()
        l = dolfin.Expression("0", degree=1) * v * dolfin.dx

        # solve problem
        T = fenics.Function(self.Vs)
        fenics.solve(a == l, T, bcs=bc)

        return T


class FEM_solution_robin:
    def __init__(self, Vs=None, param=None, meshes=None, q=None):
        self.Vs = Vs  # Location
        # self.Vs = dolfin.FunctionSpace(Vs.mesh(), 'CG', 2)  # to improve solution
        self.param = param  # Parameters
        self.meshes = meshes  # Meshes
        self.q = q  # heat source expression

    def run(self):
        # Assign boundary conditions
        boundary_parts = fenics.MeshFunction(
            "size_t", self.meshes, self.meshes.topology().dim() - 1
        )
        boundary_parts.set_all(0)  # marks whole domain as 0

        tol = 1e-8

        l_bc_point = np.min(self.Vs.mesh().coordinates()[:])
        r_bc_point = np.max(self.Vs.mesh().coordinates()[:])

        class Left(fenics.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fenics.near(x[0], l_bc_point, tol)

        class Right(fenics.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fenics.near(x[0], r_bc_point, tol)

        left = Left()
        left.mark(boundary_parts, 1)
        right = Right()
        right.mark(boundary_parts, 2)

        ds = fenics.Measure("ds", domain=self.meshes, subdomain_data=boundary_parts)

        # set up problem
        T = fenics.TrialFunction(self.Vs)
        v = fenics.TestFunction(self.Vs)

        T_amb_func = fenics.Function(self.Vs)
        T_amb_func.vector()[:] = self.param["T_amb"] * np.ones(
            len(T_amb_func.vector()[:])
        )

        F = (
            fenics.dot(fenics.grad(T), fenics.grad(v)) * fenics.dx()
            - self.q * v * fenics.dx()
            + self.param["h"] * (T - T_amb_func) * v * ds(1)
            + self.param["h"] * (T - T_amb_func) * v * ds(2)
        )
        a, l = fenics.lhs(F), fenics.rhs(F)

        # solve problem
        T = fenics.Function(self.Vs)
        fenics.solve(a == l, T)

        return T


class PGDproblem(unittest.TestCase):
    def setUp(self):
        # global parameters
        self.ord = 1  # has to be one! because of defined mapping from FD matrix Euler!

        self.ranges = [0.0, 10]  # time interval
        self.elem = 200

        self.param = {
            "T_amb": 25,
            "h": 15,
        }  # ambient temperature in °C  # heat transfer coefficient in W/(m²°C)

        # self.plotting = True
        self.plotting = False

    def TearDown(self):
        pass

    def test_solver(self):
        mesh_x = dolfin.IntervalMesh(self.elem, self.ranges[0], self.ranges[1])
        Vs_x = dolfin.FunctionSpace(mesh_x, "CG", self.ord)
        q = dolfin.Constant(1.0)

        TFD_dirichlet = FD_solution_dirichlet(
            Vs=Vs_x, meshes=mesh_x, param=self.param, q=q
        ).run()
        # print(TFD_dirichlet.compute_vertex_values()[:])

        TFD_robin = FD_solution_robin(
            Vs=Vs_x, meshes=mesh_x, param=self.param, q=q
        ).run()
        # print(TFD_robin.compute_vertex_values()[:])

        TFEM_dirichlet = FEM_solution_dirichlet(
            Vs=Vs_x, meshes=mesh_x, param=self.param, q=q
        ).run()
        # print(TFEM_dirichlet.compute_vertex_values()[:])

        TFEM_robin = FEM_solution_robin(
            Vs=Vs_x, meshes=mesh_x, param=self.param, q=q
        ).run()
        # print(TFEM_robin.compute_vertex_values()[:])

        # check errors
        error_dirichlet = dolfin.errornorm(TFD_dirichlet, TFEM_dirichlet)
        print("error dirichlet bc FD - FEM:", error_dirichlet)

        error_robin = dolfin.errornorm(TFD_robin, TFEM_robin)
        print("error robin bc FD - FEM:", error_robin)

        self.assertTrue(error_dirichlet < 1e-8)
        self.assertTrue(
            error_robin < 1e-1
        )  # expect up to single percent error for robin bc

        if self.plotting:
            import matplotlib.pyplot as plt

            # dirichlet bc
            plt.figure()
            plt.plot(
                mesh_x.coordinates()[:],
                TFD_dirichlet.compute_vertex_values()[:],
                "-*b",
                label="FD",
            )
            plt.plot(
                mesh_x.coordinates()[:],
                TFEM_dirichlet.compute_vertex_values()[:],
                "-*g",
                label="FEM",
            )
            plt.legend()
            plt.title("Dirichlet boundary condition")
            plt.xlabel("space x")
            plt.ylabel("Temperature T in °C")

            # robin bc
            plt.figure()
            plt.plot(
                mesh_x.coordinates()[:],
                TFD_robin.compute_vertex_values()[:],
                "-*b",
                label="FD",
            )
            plt.plot(
                mesh_x.coordinates()[:],
                TFEM_robin.compute_vertex_values()[:],
                "-*g",
                label="FEM",
            )
            plt.legend()
            plt.title("Robin boundary condition")
            plt.xlabel("space x")
            plt.ylabel("Temperature T in °C")

            # error robin bc
            plt.figure()
            plt.plot(
                mesh_x.coordinates()[:],
                100
                * (
                    TFEM_robin.compute_vertex_values()[:]
                    - TFD_robin.compute_vertex_values()[:]
                )
                / TFEM_robin.compute_vertex_values()[:],
                "-*b",
                label="error FD - FEM",
            )
            plt.legend()
            plt.title("Relative error for robin boundary condition")
            plt.xlabel("space x")
            plt.ylabel("relative error in %")

            plt.show()


if __name__ == "__main__":
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging

    logging.basicConfig(level=logging.INFO)

    unittest.main()
