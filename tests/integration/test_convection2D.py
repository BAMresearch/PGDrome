'''
    2D transient thermo problem with convection boundary condition
    PGD variables: space: x, time: t, convection parameter h

...


'''

import unittest
import dolfin
import numpy as np

from pgdrome.solver import PGDProblem, FD_matrices

def create_meshes(num_elem, ord, ranges):

    meshes = list()
    Vs = list()

    dim = len(num_elem)

    for i in range(dim):
        mesh_tmp = dolfin.IntervalMesh(num_elem[i], ranges[i][0], ranges[i][1])
        Vs_tmp = dolfin.FunctionSpace(mesh_tmp, 'CG', ord[i])

        meshes.append(mesh_tmp)
        Vs.append(Vs_tmp)

    return meshes, Vs

def create_doms_fem(v,param,fact_dimless):
    boundary = dolfin.MeshFunction("size_t", v.mesh(), v.mesh().topology().dim()-1)
    boundary.set_all(0)

    class Bottom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[1], 0., dolfin.DOLFIN_EPS)

    class Top(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[1], param['ly']/fact_dimless['y_0'], dolfin.DOLFIN_EPS)

    class Left(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], 0., dolfin.DOLFIN_EPS)

    class Right(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], param['lx']/fact_dimless['x_0'], dolfin.DOLFIN_EPS)

    bottom = Bottom()
    bottom.mark(boundary, 1)  # marks bottom of the domain as 1
    top = Top()
    top.mark(boundary, 2)  # marks top of the domain as 2
    left = Left()
    left.mark(boundary, 3)  # marks left side of the domain as 3
    right = Right()
    right.mark(boundary, 4)  # marks right side of the domain as 4

    return boundary

# reference model FEM in space and backward euler in time
class Reference:

    def __init__(self, param={}, vs=[], dom=None, point=None):

        self.vs = vs  # Location
        self.param = param  # Parameters

        # time points
        self.time_mesh = self.vs[1].mesh().coordinates()[:]
        self.T_n = dolfin.interpolate(self.param['IC_t'], self.vs[0])
        self.T_amb = dolfin.interpolate(self.param['Tamb_fct'], self.vs[0])

        # problem
        self.mesh = self.vs[0].mesh()
        ds = dolfin.Measure('ds', domain=self.mesh, subdomain_data=dom)
        T = dolfin.TrialFunction(self.vs[0])
        v = dolfin.TestFunction(self.vs[0])
        self.dt = dolfin.Constant(1.)
        self.h = dolfin.Constant(1.)
        # splitted dot(grad,grad) Term for dimless
        self.F = self.param['a1'] * self.param["rho"] * self.param["cp"] * T * v * dolfin.dx() \
                 + self.dt * self.param['ax'] * self.param["k"] * T.dx(0) * v.dx(0) * dolfin.dx() + self.dt * self.param['ay'] * self.param["k"] * T.dx(1) * v.dx(1) * dolfin.dx() \
                 + self.dt * self.param['hy'] * self.h * (T - self.T_amb) * v * ds(2) \
                 + self.dt * self.param['hx'] * self.h * (T - self.T_amb) * v * ds(4) \
                 - self.param['a1'] * self.param["rho"] * self.param["cp"] * self.T_n * v * dolfin.dx()
                # + self.dt * self.param['hy'] * self.h * (T - self.T_amb) * v * ds(1) \
                # + self.dt * self.param['hx'] * self.h * (T - self.T_amb) * v * ds(3) \ # more convection boundaries

        # without split
        # self.F = self.param["rho"] * self.param["cp"] * T * v * dolfin.dx() \
        #          + self.dt * self.param["k"] *  dolfin.dot(dolfin.grad(T), dolfin.grad(v)) * dolfin.dx() \
        #          + self.dt * self.h * (T - self.T_amb) * v * ds(2) \
        #          + self.dt * self.h * (T - self.T_amb) * v * ds(4) \
        #          - self.param["rho"] * self.param["cp"] * self.T_n * v * dolfin.dx()
        #        # + self.dt * self.h * (T - self.T_amb) * v * ds(1) \
        #        # + self.dt * self.h * (T - self.T_amb) * v * ds(3) \ # more convection boundaries


        self.point = point

    def __call__(self, values):

        # check time mesh for requested time value
        if not np.where(self.time_mesh == values[0])[0]:
            print("ERROR time step not in mesh What to do?")
        self.h.assign(values[1]) # * self.param["h0"]) #?

        # Time-stepping
        Ttime = []
        Ttmp = dolfin.Function(self.vs[0])
        Ttmp.vector()[:] = 1 * self.T_n.vector()[:]
        Ttime.append(Ttmp)  # otherwise it will be overwritten with new solution
        Tpoint = [np.copy(self.T_n(self.point))]
        T = dolfin.Function(self.vs[0])
        for i in range(len(self.time_mesh) - 1):
            self.dt.assign(self.time_mesh[i + 1] - self.time_mesh[i])
            # Compute solution
            a, L = dolfin.lhs(self.F), dolfin.rhs(self.F)
            dolfin.solve(a == L, T)

            # Update previous solution
            self.T_n.assign(T)

            # store solution
            Ttmp = dolfin.Function(self.vs[0])
            Ttmp.vector()[:] = 1 * T.vector()[:]
            Ttime.append(Ttmp)
            Tpoint.append(np.copy(T(self.point)))

        return Ttime, Tpoint  # solution in time over xy and time solution at fixed point (x,y)


# test problem
class problem(unittest.TestCase):

    def setUp(self):
        # global parameters
        self.param = {"rho": 7100, "cp": 3100, "k": 1000, 'Tamb': 25, 'T_ic': 500,
                      'lx': 0.15, 'ly': 0.1, 'lt': 100}

        # possiblity to make the equation dimless
        # self.factors_o = {'x_0': 0.15, 'y_0': 0.1, 't_0': 100., 'T_0': 500.}
        self.factors_o = {'x_0': 1, 'y_0': 1, 't_0': 1., 'T_0': 1.}

        # factors for pde
        self.param['a1'] = self.factors_o['T_0'] / self.factors_o['t_0']
        self.param['ax'] = self.factors_o['T_0'] / self.factors_o['x_0'] ** 2
        self.param['ay'] = self.factors_o['T_0'] / self.factors_o['y_0'] ** 2
        self.param['hx'] = self.factors_o['T_0'] / self.factors_o['x_0']
        self.param['hy'] = self.factors_o['T_0'] / self.factors_o['y_0']

        self.ranges = [[0., self.param['lx']/ self.factors_o['x_0']],  # xmin, xmax
                       [0., self.param['ly']/ self.factors_o['y_0']],  # ymin, ymax
                       [0., self.param['lt']/ self.factors_o['t_0']],  # tmin, tmax
                       [2, 200]]  # hmin, hmax

        self.ords = [1, 1, 1, 1]  # x, y, t, h
        self.elems = [10, 10, 20, 50]

        self.h_fixed = 200
        self.point = (self.param['lx'] / self.factors_o['x_0'], self.param['ly']/(2*self.factors_o['y_0']))

        # self.plotting = True
        self.plotting = False

    def TearDown(self):
        pass

    def test_reference(self):
        self.param['Tamb_fct'] = dolfin.Expression('Tamb', degree=1, Tamb=self.param["Tamb"]/self.factors_o['T_0'])  # ambient condition FEM
        self.param['IC_t'] = dolfin.Expression('T', degree=1, T=self.param["T_ic"]/self.factors_o['T_0'])  # initial condition FEM

        # PGD meshes:
        meshes, vs = create_meshes(self.elems, self.ords, self.ranges)

        # FEM meshes
        mesh_xy = dolfin.RectangleMesh(dolfin.Point(self.ranges[0][0],self.ranges[1][0]),dolfin.Point(self.ranges[0][1],self.ranges[1][1]), self.elems[0], self.elems[1])
        v_xy = dolfin.FunctionSpace(mesh_xy,'CG',2)
        v_t = vs[2]
        doms = create_doms_fem(v_xy,self.param, self.factors_o)

        u_fem, u_fem2 = Reference(param=self.param, vs=[v_xy, v_t], dom = doms, point=self.point)(
            [self.param['lt'], self.h_fixed])
        print(np.array(u_fem2)*self.factors_o['T_0'])

        if self.plotting:
            import matplotlib.pyplot as plt

            plt.figure(1)
            plt.plot(meshes[2].coordinates()[:]* self.factors_o['t_0'], np.array(u_fem2)*self.factors_o['T_0'], '-or', label='FEM')
            plt.title(f"solution at [(x,y),h]={self.point},{self.h_fixed} over time")
            plt.xlabel("time t [m]")
            plt.ylabel("Temperature T [°C]")
            plt.legend()

            plt.figure(2)
            dolfin.plot(u_fem[-1])
            plt.show()

        self.assertTrue(0.00001 < 1e-3)

if __name__ == '__main__':
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging
    logging.basicConfig(level=logging.INFO)

    unittest.main()