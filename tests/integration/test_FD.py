'''
    check FD formulation working within fenics

    problem rho cp \partial{T}{dt} = q(t)
    solve for T
'''

import unittest
import dolfin
import fenics
import numpy as np

from scipy.sparse import spdiags

def FD_matrices(x):
    N = len(x)
    e = np.ones(N)

    M = spdiags(e, 0, N, N).toarray()
    D2 = spdiags([e, e, e], [-1, 0, 1], N, N).toarray()
    D1_up = spdiags([e, e], [-1, 0], N, N).toarray()

    i = 0
    hp = x[i + 1] - x[i]

    M[i, i] = hp / 2

    D2[i, i] = -1 / hp
    D2[i, i + 1] = 1 / hp

    D1_up[i, i] = -1 / 2
    D1_up[i, i + 1] = 1 / 2

    for i in range(1, N - 1, 1):
        hp = x[i + 1] - x[i]
        hm = x[i] - x[i - 1]

        M[i, i] = (hp + hm) / 2

        D2[i, i] = -(hp + hm) / (hp * hm)
        D2[i, i + 1] = 1 / hp
        D2[i, i - 1] = 1 / hm

        D1_up[i, i] = (hp + hm) / (2 * hm)
        D1_up[i, i - 1] = -(hp + hm) / (2 * hm)

    i = N - 1
    hm = x[i] - x[i - 1]

    M[i, i] = hm / 2

    D2[i, i] = -1 / hm
    D2[i, i - 1] = 1 / hm

    D1_up[i, i] = (hp + hm) / (2 * hm)
    D1_up[i, i - 1] = -(hp + hm) / (2 * hm)
    return M, D2, D1_up

class ref_solution():

    def __init__(self, Vs=None, param=None, meshes=None, q=None):

        self.Vs = Vs  # Location
        self.param = param  # Parameters
        self.meshes = meshes  # Meshes
        self.q = q # heat source expression

    def run(self):
        T = dolfin.Function(self.Vs)

        T_tmp = np.zeros(len(T.vector()[:]))
        T_tmp[0] = self.param['T_amb']

        idx_sort = np.argsort(self.Vs.tabulate_dof_coordinates()[:].flatten())
        time_points = np.sort(self.Vs.tabulate_dof_coordinates()[:].flatten())

        for i in range(1,len(time_points)):
            T_tmp[i] = T_tmp[i-1] + \
                       (time_points[i]-time_points[i-1])/(self.param["rho"]*self.param["c_p"])\
                       *self.q(time_points[i])


        T.vector()[:] = T_tmp[idx_sort]

        return T


class FD_solution():

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
        Q = dolfin.interpolate(self.q,self.Vs).vector()[:]

        # set up initial condition
        IC = np.zeros(len(Q))
        IC[-1] = self.param['T_amb']
        
        # set up shorted heat equation
        Amat = self.param['rho']*self.param['c_p']*D1_up
        Fvec = M1 @ Q - self.param['rho']*self.param['c_p'] * D1_up @ IC
        
        # set matrices for initial condition
        Fvec[-1] = 0
        Amat[:,-1] = 0
        Amat[-1,:] = 0
        Amat[-1,-1] = 1

        # solve problem
        vec_tmp = np.linalg.solve(Amat,Fvec)

        # add inital condition
        T.vector()[:] = vec_tmp + IC

        return T


class FEM_solution():

    def __init__(self, Vs=None, param=None, meshes=None, q=None):

        self.Vs = Vs  # Location
        self.param = param  # Parameters
        self.meshes = meshes  # Meshes
        self.q = q # heat source expression

    def run(self):

        #boundary condition
        def init(x, on_boundary):
            return x < 0.0 + 1E-5

        bc = dolfin.DirichletBC(self.Vs, self.param["T_amb"], init)

        # set up functions
        T = fenics.TrialFunction(self.Vs)
        v = fenics.TestFunction(self.Vs)

        # set up problem
        a = self.param["rho"]*self.param["c_p"]*T.dx(0)*v*dolfin.dx
        l = self.q*v*dolfin.dx

        # solve problem
        T = fenics.Function(self.Vs)
        fenics.solve(a == l, T, bcs=bc)

        # # add inital condition
        # T.vector()[-1] = self.param["T_amb"]

        return T
    
    
class PGDproblem(unittest.TestCase):

    def setUp(self):
        # global parameters
        self.ord = 1

        self.ranges = [0., 50] # time intervall
        self.elem = 200

        self.param = {"rho": 71, "c_p": 31, "k": 100, 'P':250, 'T_amb':25} # material density [kg/m³]  # heat conductivity [W/m°C] # specific heat capacity [J/kg°C]

    def TearDown(self):
        pass

    def test_solver(self):
        mesh_t = dolfin.IntervalMesh(self.elem, self.ranges[0], self.ranges[1])
        Vs_t = dolfin.FunctionSpace(mesh_t, 'CG', self.ord)
        q = dolfin.Expression('x[0] < 5 ? 0 : (x[0] > 20 ? 0 : Q)', degree=1, Q=self.param['P'])

        # reference backward euler
        Tref = ref_solution(Vs=Vs_t, meshes=mesh_t, param=self.param, q=q).run()
        print(Tref.compute_vertex_values()[:])

        TFD = FD_solution(Vs=Vs_t, meshes=mesh_t, param=self.param, q=q).run()
        print(TFD.compute_vertex_values()[:])
        
        TFEM = FEM_solution(Vs=Vs_t, meshes=mesh_t, param=self.param, q=q).run()
        print(TFEM.compute_vertex_values()[:])

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(mesh_t.coordinates()[:],Tref.compute_vertex_values()[:],'-*r', label='ref')
        plt.plot(mesh_t.coordinates()[:], TFD.compute_vertex_values()[:], '-*b', label='FD')
        plt.plot(mesh_t.coordinates()[:], TFEM.compute_vertex_values()[:], '-*g', label='FEM')
        plt.legend()
        plt.show()




if __name__ == '__main__':
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging
    logging.basicConfig(level=logging.INFO)

    unittest.main()