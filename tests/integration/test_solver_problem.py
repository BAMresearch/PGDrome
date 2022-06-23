r'''
    2D linear elastictity PGD example
        geometry: 1000 x 100
        boundary: fixed at left side
        load: at top: first half with 1.5xloadfactor second half 0.5xloadfactor
        elastic plane strain C(E,nu)
          ||||F2||||||F1|||
        |>-----------------
        |>----------------- Ly
                Lx
        with E = lam_E E_0 and F2=lam_p F20/ F1=lam_p F10
    PGD for displacements with PGD variable: X (x,y space), lam_p (load factor), lam_E (E Module factor), nu (Poission ratio)
    DGL: \int var_eps C(E,nu) eps dX = \int var_u F2 dX_F2 + \int var_u F1 dX_F1
'''

import unittest
import dolfin
import os
import numpy as np

from pgdrome.solver import PGDProblem1
from pgdrome.model import PGDErrorComputation

dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = 'uflacs'


def create_meshesExtra(num_elem, ord, ranges):
    '''
    :param num_elem: list for each extra PG CO
    :param ord: list for each extra PG CO
    :param ranges: list for each extra PG CO
    :return: meshes and V
    '''

    meshes = list()
    Vs = list()

    dim = len(num_elem)
    if dim != len(ord):
        print('len(num_elem) %s != len(ord) %s' %(dim. len(ord)))
        raise ValueError('lenght of input not equal!!')

    for i in range(dim):
        mesh_tmp = dolfin.IntervalMesh(num_elem[i], ranges[i][0], ranges[i][1])
        Vs_tmp = dolfin.FunctionSpace(mesh_tmp, 'P', ord[i])

        meshes.append(mesh_tmp)
        Vs.append(Vs_tmp)

    return meshes, Vs


def create_meshX(N, order):
    '''
        create mesh and boundaries in space X
    :param L: dimensions Lx,Ly
    :param N: number of elements in x,y
    :param order: order for function space
    :return: mesh, boundaries, V
    '''
    '''mesh'''
    L = [1000,100] # LX, LY
    mesh_x = dolfin.RectangleMesh(dolfin.Point(0., 0.), dolfin.Point(L[0], L[1]), N[0], N[1], "crossed")
    V_x = dolfin.VectorFunctionSpace(mesh_x, 'P', order)  # !!! VECTOR FUNCTION SPACE

    return mesh_x, V_x


def create_dom(Vs,param):
    # boundaries in x
    subdomains = dolfin.MeshFunction("size_t", Vs[0].mesh(), Vs[0].mesh().topology().dim())  # same dimension
    boundaries = dolfin.MeshFunction("size_t", Vs[0].mesh(), Vs[0].mesh().topology().dim() - 1)  # dim-1 -> surfaces
    subdomains.set_all(0)
    boundaries.set_all(0)

    L = [1000, 100]  # LX, LY

    class left(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return dolfin.near(x[0], 0.)

    class top_left(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return dolfin.near(x[1], L[1]) and x[0] < 0.5 * L[0]

    class top_right(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return dolfin.near(x[1], L[1]) and x[0] > 0.5 * L[0]

    class right(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return dolfin.near(x[0], L[0])

    Left = left()
    Left.mark(boundaries, 1)
    Top_left = top_left()
    Top_left.mark(boundaries, 2)
    Top_right = top_right()
    Top_right.mark(boundaries, 3)
    Right = right()
    Right.mark(boundaries, 4)

    return [boundaries,0,0,0]


def create_bc(Vs,dom,param):
    # create boundary condition list
    boundaries = dom[0]

    bc_x = [dolfin.DirichletBC(Vs[0], dolfin.Constant((0., 0.)), boundaries, 1)]

    return [bc_x, 0, 0, 0]


def problem_assemble_lhs(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem
    # ds = Measure('ds', domain=meshes[0], subdomain_data=dom[0])

    def eps(v):
        """ Returns a vector of strains of size (3,1) in the Voigt notation
        layout {eps_xx, eps_yy, gamma_xy} where gamma_xy = 2*eps_xy"""
        return dolfin.as_vector([v[i].dx(i) for i in range(2)] +
                                [v[i].dx(j) + v[j].dx(i) for (i, j) in [(0, 1)]])

    if typ == 'r':
        a = dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][0] * Fs[3] * dolfin.dx(meshes[3])) )\
            * dolfin.inner(param["Cmatrix"][0]*eps(fct_F),eps(var_F)) * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][1] * Fs[3] * dolfin.dx(meshes[3])) )\
            * dolfin.inner(param["Cmatrix"][1]*eps(fct_F),eps(var_F)) * dolfin.dx(meshes[0])

    if typ == 's':
        a = dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][0] * Fs[3] * dolfin.dx(meshes[3])) )\
            * var_F * fct_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][1] * Fs[3] * dolfin.dx(meshes[3])) )\
            * var_F * fct_F * dolfin.dx(meshes[1])

    if typ == 't':
        a = dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][0] * Fs[3] * dolfin.dx(meshes[3])) )\
            * var_F * param["E_func"] * param["E_0"] * fct_F * dolfin.dx(meshes[2]) \
            + dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][1] * Fs[3] * dolfin.dx(meshes[3])) )\
            * var_F * param["E_func"] * param["E_0"] * fct_F * dolfin.dx(meshes[2])

    if typ == 'v':
        a = dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) )\
            * var_F * param["Nu_func"][0] * fct_F * dolfin.dx(meshes[3]) \
            + dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) )\
            * var_F * param["Nu_func"][1] * fct_F * dolfin.dx(meshes[3])

    return a

def problem_assemble_rhs(fct_F,var_F,Fs,meshes,dom,param,G,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem
    ds = dolfin.Measure('ds', domain=meshes[0], subdomain_data=dom[0])

    def eps(v):
        """ Returns a vector of strains of size (3,1) in the Voigt notation
        layout {eps_xx, eps_yy, gamma_xy} where gamma_xy = 2*eps_xy"""
        return dolfin.as_vector([v[i].dx(i) for i in range(2)] +
                                [v[i].dx(j) + v[j].dx(i) for (i, j) in [(0, 1)]])

    if typ == 'r':
        l = 0
        for ext in range(len(G[0][0])):

            l += dolfin.Constant(dolfin.assemble(Fs[1] * G[0][1][ext] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * G[0][2][ext] * dolfin.dx(meshes[2])) \
                * dolfin.assemble(Fs[3] * G[0][3][ext] * dolfin.dx(meshes[3])) )\
                * dolfin.dot(G[0][0][ext],var_F) * ds(2) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * G[1][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * G[1][2][ext] * dolfin.dx(meshes[2])) \
                 * dolfin.assemble(Fs[3] * G[1][3][ext] * dolfin.dx(meshes[3])) )\
                 * dolfin.dot(G[1][0][ext],var_F) * ds(3)

        if nE > 0:
            for old in range(nE):
                l += -dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][0] * PGD_func[3][old] * dolfin.dx(meshes[3])) )\
                    * dolfin.inner(param["Cmatrix"][0]*eps(PGD_func[0][old]),eps(var_F)) * dolfin.dx(meshes[0]) \
                    - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][1] * PGD_func[3][old] * dolfin.dx(meshes[3]))) \
                    * dolfin.inner(param["Cmatrix"][1]*eps(PGD_func[0][old]),eps(var_F)) * dolfin.dx(meshes[0])

    if typ == 's':
        l = 0
        for ext in range(len(G[0][1])):
            l += dolfin.Constant(dolfin.assemble(dolfin.dot(G[0][0][ext],Fs[0])*ds(2)) \
                 * dolfin.assemble(Fs[2] * G[0][2][ext] * dolfin.dx(meshes[2])) \
                 * dolfin.assemble(Fs[3] * G[0][3][ext] * dolfin.dx(meshes[3])) )\
                 * var_F * G[0][1][ext] * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(dolfin.dot(G[1][0][ext],Fs[0])*ds(3)) \
                 * dolfin.assemble(Fs[2] * G[1][2][ext] * dolfin.dx(meshes[2])) \
                 * dolfin.assemble(Fs[3] * G[1][3][ext] * dolfin.dx(meshes[3])) )\
                 * var_F * G[1][1][ext] * dolfin.dx(meshes[1])
        if nE > 0:
            for old in range(nE):
                l += -dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][0] * PGD_func[3][old] * dolfin.dx(meshes[3])) )\
                    * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                    - dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][1] * PGD_func[3][old] * dolfin.dx(meshes[3])) )\
                    * var_F * PGD_func[1][old] * dolfin.dx(meshes[1])

    if typ == 't':
        l = 0
        for ext in range(len(G[0][1])):
            l += dolfin.Constant(dolfin.assemble(dolfin.dot(G[0][0][ext],Fs[0])*ds(2)) \
                 * dolfin.assemble(Fs[1] * G[0][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[3] * G[0][3][ext] * dolfin.dx(meshes[3])) )\
                 * var_F * G[0][2][ext] * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(dolfin.dot(G[1][0][ext],Fs[0])*ds(3)) \
                 * dolfin.assemble(Fs[1] * G[1][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[3] * G[1][3][ext] * dolfin.dx(meshes[3]))) \
                 * var_F * G[1][2][ext] * dolfin.dx(meshes[2])
        if nE > 0:
            for old in range(nE):
                l += -dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][0] * PGD_func[3][old] * dolfin.dx(meshes[3])) )\
                    * var_F * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                    - dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][1] * PGD_func[3][old] * dolfin.dx(meshes[3])) )\
                    * var_F * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])

    if typ == 'v':
        l = 0
        for ext in range(len(G[0][1])):
            l += dolfin.Constant(dolfin.assemble(dolfin.dot(G[0][0][ext],Fs[0])*ds(2)) \
                 * dolfin.assemble(Fs[1] * G[0][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * G[0][2][ext] * dolfin.dx(meshes[2]))) \
                 * var_F * G[0][3][ext] * dolfin.dx(meshes[3]) \
                + dolfin.Constant(dolfin.assemble(dolfin.dot(G[1][0][ext],Fs[0])*ds(3)) \
                 * dolfin.assemble(Fs[1] * G[1][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * G[1][2][ext] * dolfin.dx(meshes[2])) )\
                 * var_F * G[1][3][ext] * dolfin.dx(meshes[3])
        if nE > 0:
            for old in range(nE):
                l += -dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                    * var_F * param["Nu_func"][0] * PGD_func[3][old] * dolfin.dx(meshes[3]) \
                    - dolfin.Constant(dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) )\
                    * var_F * param["Nu_func"][1] * PGD_func[3][old] * dolfin.dx(meshes[3])

    return l

def main_normal(vs, params, writeFlag=False, name='PGDsolution', problem='linear', settings={"linear_solver":"mumps"}):
    '''
        computation of PGD solution for given problem normal
        :param vs: list of function spaces len = num_pgd_var
        :param writeFlag: save files or not
        :return: PGDModel and PGDProblem1
    '''

    PGD_nmax = 7  # PGD enrichment number (max number of PGD modes per coordinate)
    prob = ['r', 's', 't', 'v']  # Reihenfolge !! PGD_dim
    seq_fp = [0, 1, 2, 3]  # default reihenfolge => prob

    # define some parameter
    efunc = dolfin.Expression('x[0]', degree=4)
    # decomposition plane stress and plane strain
    c1_np = np.array([[1.0, 1.0, 0.0],
                      [1.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0]])
    c2_np = np.array([[1.0, -1.0, 0.0],
                      [-1.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
    c1 = dolfin.as_matrix(c1_np)
    c2 = dolfin.as_matrix(c2_np)

    # decomposition for plane stress
    # nu_func1 = dolfin.Expression("1.0/(2.0 * (1.0 - x[0]))",degree=10)
    # nu_func2 = dolfin.Expression("1.0/(2.0 * (1.0 + x[0]))",degree=10)
    # decomposition for plane strain
    nu_func1 = dolfin.Expression("1.0/(2.0 * (1.0 + x[0]) * (1.0 - 2.0 * x[0]))", degree=10)
    nu_func2 = dolfin.Expression("1.0/(2.0 * (1.0 + x[0]))", degree=10)

    # put into param dict
    params["Nu_func"] = [nu_func1, nu_func2]
    params["Cmatrix"] = [c1, c2]
    params["E_func"] = efunc

    # define separated load expressions
    g1_1 = [params['g1']]
    g1_2 = [dolfin.Expression('x[0]', degree=4)]
    g1_3 = [dolfin.Expression('1.0', degree=4)]
    g1_4 = [dolfin.Expression('1.0', degree=4)]

    g2_1 = [params['g2']]
    g2_2 = [dolfin.Expression('x[0]', degree=4)]
    g2_3 = [dolfin.Expression('1.0', degree=4)]
    g2_4 = [dolfin.Expression('1.0', degree=4)]

    load = [[g1_1, g1_2, g1_3, g1_4], [g2_1, g2_2, g2_3, g2_4]]

    # PGDProblem class
    pgd_prob = PGDProblem1(name='PGD_xpEv', name_coord=['X', 'P', 'E', 'nu'],
                           modes_info=['U', 'Node', 'Vector'],
                           Vs=vs, bc_fct=create_bc, load=load,
                           param=params, dom_fct=create_dom,
                           rhs_fct=problem_assemble_rhs, lhs_fct=problem_assemble_lhs,
                           probs=prob, seq_fp=seq_fp, PGD_nmax=PGD_nmax)

    # solve displacement problem
    # pgd_prob.max_fp_it = 15
    # pgd_prob.stop_fp = 'norm' #'delta'
    # pgd_prob.tol_fp_it = 1e-8
    # pgd_prob.tol_abs = 1e-4
    pgd_prob.solve_PGD(_problem=problem,settings=settings)  # solve


    # print('computed:', pgd_prob.name, pgd_prob.amplitude)
    print(pgd_prob.simulation_info)

    pgd_solution = pgd_prob.return_PGD()

    # pgd_solution.print_info()

    # save for postprocessing - paraview!!
    if writeFlag:
        folder = os.path.join(os.path.dirname(__file__), '..', 'results', name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        pgd_solution.write_hdf5(folder)
        pgd_solution.write_pxdmf(folder)

    return pgd_prob, pgd_solution

class FEM_reference():
    '''FEM reference model'''
    def __init__(self,V_x,params,x_values=[]):

        self.V_x = V_x
        self.params = params
        self.mesh = self.V_x.mesh()

        self.x_values = x_values

        # set up problem
        doms = create_dom([self.V_x], self.params)
        self.bc_x = create_bc([self.V_x], doms, self.params)[0]
        self.ds = dolfin.Measure('ds', domain=self.mesh, subdomain_data=doms[0])

    def eps(self,v):
        """ Returns a vector of strains of size (3,1) in the Voigt notation
        layout {eps_xx, eps_yy, gamma_xy} where gamma_xy = 2*eps_xy"""
        return dolfin.as_vector([v[i].dx(i) for i in range(2)] +
                            [v[i].dx(j) + v[j].dx(i) for (i, j) in [(0, 1)]])

    def sigma(self,v,E,nu):
        # plane strain in E and nu
        alpha = E / ((1. + nu) * (1 - 2.0 * nu))
        C_np = alpha * np.array([[1.0 - nu, nu, 0.0],
                                    [nu, 1.0 - nu, 0.0],
                                    [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]])
        C = dolfin.as_matrix(C_np)
        return C*self.eps(v)

    def __call__(self,values):
        # set values
        lam_p = values[0]
        lam_E = values[1]
        nu = values[2]
        E = lam_E * self.params['E_0']

        #solve problem
        v = dolfin.TestFunction(self.V_x)
        u = dolfin.TrialFunction(self.V_x)
        a = dolfin.inner(self.sigma(u,E,nu), self.eps(v)) * dolfin.dx
        l = lam_p* dolfin.dot(self.params['g1'], v) * self.ds(2) + lam_p*dolfin.dot(self.params['g2'], v) * self.ds(3)
        u = dolfin.Function(self.V_x, name="Displacement")
        dolfin.solve(a == l, u, self.bc_x)

        # if specific points are given
        if self.x_values:
            u_out = np.zeros((len(self.x_values),2))
            for i in range(len(self.x_values)):
                u_out[i,:]=np.array(u(self.x_values[i]))
            return u_out
        else:
            # return full vector
            return u


class TestSolverProblem(unittest.TestCase):

    def setUp(self):
        # global parameters
        self.ords = [2, 1, 1, 1] # order for each mesh
        self.ranges = [[0., 2.],  # pmin, pmax
                  [0.5, 1.5],  # Emin,Emax
                  [0.1, 0.4]]  # numin,numax
        self.numElems = [2, 50, 50] # p, E, nu
        self.params = {"E_0": 30000, "g1":dolfin.Constant((0., -0.5)), "g2":dolfin.Constant((0., -1.5))}

        self.write = False # set to True to save pxdmf file
        # self.write = True

        # test values
        self.p = 1.5
        self.E = 0.75
        self.nu = 0.2
        self.x = (1000/2,100/2) # middle

    def TearDown(self):
        pass

    def test_solver_options(self):
        # define meshes
        # mesh in x space
        _, v_x = create_meshX([200, 20], self.ords[0])
        _, v_e = create_meshesExtra(self.numElems, self.ords[1:4], self.ranges)
        # solve PGD problem with linear solver
        pgd_prob_lin, pgd_s_lin = main_normal([v_x] + v_e, self.params, writeFlag=self.write, name='PGDsolution', problem='linear')
        # solve PGD problem with nonlinear solver
        pgd_prob_nl, pgd_s_nl = main_normal([v_x] + v_e, self.params, writeFlag=self.write, name='PGDsolution', problem='nonlinear', settings={"relative_tolerance":1e-8, "linear_solver": "mumps"})

        # check solver convergences
        print('PGD amplitudes', pgd_prob_lin.amplitude, pgd_prob_nl.amplitude)
        amplitude_diff_max = (np.array(pgd_prob_lin.amplitude) - np.array(pgd_prob_nl.amplitude)).max()
        print('diff amplitudes',amplitude_diff_max)
        self.assertTrue(amplitude_diff_max < 1e-8)

        ### Error checks
        # error to FEM at one point in x and specified PGD parameters (for linear solver problem)
        ref_fem = FEM_reference(v_x,self.params)
        pgd_u = pgd_s_lin.evaluate(0, [1, 2, 3], [self.p, self.E, self.nu], 0)
        ref_u = ref_fem([self.p,self.E,self.nu])
        print('ref_u', ref_u(self.x),'pgd_u',pgd_u(self.x))
        error_point = np.linalg.norm(np.array(pgd_u(self.x)-ref_u(self.x)))/np.linalg.norm(ref_u(self.x))
        # errornorm over all nodes for given set of PGD variables
        errorL2 = np.linalg.norm(pgd_u.compute_vertex_values()[:] - ref_u.compute_vertex_values()[:], 2) / np.linalg.norm(ref_u.compute_vertex_values()[:], 2)
        print(error_point, errorL2)

        # check
        self.assertTrue(error_point < pgd_prob_lin.amplitude[-2]) # 1e-4
        self.assertTrue(errorL2 < pgd_prob_lin.amplitude[-2])

        # check PGDErrorComputation class with manually computation
        # over space x at given PGD coordinates
        error_class = PGDErrorComputation(fixed_dim=[0],
                                          data_test=[[self.p, self.E, self.nu]],
                                          FOM_model=ref_fem,
                                          PGD_model=pgd_s_lin)
        
        errors, mean_errorL2, max_errorL2 = error_class.evaluate_error()  # Computing Error
        print(errors, mean_errorL2, max_errorL2)
        self.assertAlmostEqual(max_errorL2, errorL2, places=8)
        
        # # specified point
        ref_fem.x_values=[self.x]
        print(ref_fem([self.p,self.E,self.nu]))
        error_class.fixed_var=[self.x]
        errors, mean_errorL2, max_errorL2 = error_class.evaluate_error()  # Computing Error
        print(errors, mean_errorL2, max_errorL2)
        self.assertAlmostEqual(max_errorL2, error_point, places=8)

        # suggested way to check solution:
        # error to fem computation at random values for PGD variables over space x
        ref_fem.x_values = [] # delete specified x values
        error_random = PGDErrorComputation(fixed_dim=[0], n_samples=10, FOM_model=ref_fem, PGD_model=pgd_s_lin)
        _, mean_errorL2, max_errorL2 = error_random.evaluate_error()  # Computing Error
        print(mean_errorL2, max_errorL2)
        self.assertTrue(mean_errorL2 < pgd_prob_lin.amplitude[-2])
        self.assertTrue(max_errorL2 < pgd_prob_lin.amplitude[-2])



if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=logging.DEBUG)

    unittest.main()

    unittest.main()
