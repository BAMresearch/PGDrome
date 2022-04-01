'''
    roadway crossection 2D  PGD SOLUTION WITH FENICS

    PGD variable: x, lam_p, lam_E, nu

'''


import sys
import numpy as np
import os
import dolfin

from pgdrome.solver import PGDProblem1

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
        a = dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][0] * Fs[3] * dolfin.dx(meshes[3])) \
            * dolfin.inner(param["Cmatrix"][0]*eps(fct_F),eps(var_F)) * dolfin.dx(meshes[0]) \
            + dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][1] * Fs[3] * dolfin.dx(meshes[3])) \
            * dolfin.inner(param["Cmatrix"][1]*eps(fct_F),eps(var_F)) * dolfin.dx(meshes[0])

    if typ == 's':
        a = dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][0] * Fs[3] * dolfin.dx(meshes[3])) \
            * var_F * fct_F * dolfin.dx(meshes[1]) \
            + dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][1] * Fs[3] * dolfin.dx(meshes[3])) \
            * var_F * fct_F * dolfin.dx(meshes[1])

    if typ == 't':
        a = dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][0] * Fs[3] * dolfin.dx(meshes[3])) \
            * var_F * param["E_func"] * param["E_0"] * fct_F * dolfin.dx(meshes[2]) \
            + dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[3] * param["Nu_func"][1] * Fs[3] * dolfin.dx(meshes[3])) \
            * var_F * param["E_func"] * param["E_0"] * fct_F * dolfin.dx(meshes[2])

    if typ == 'v':
        a = dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
            * var_F * param["Nu_func"][0] * fct_F * dolfin.dx(meshes[3]) \
            + dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(Fs[0]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * Fs[2] * dolfin.dx(meshes[2])) \
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

            l += dolfin.assemble(Fs[1] * G[0][1][ext] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * G[0][2][ext] * dolfin.dx(meshes[2])) \
                * dolfin.assemble(Fs[3] * G[0][3][ext] * dolfin.dx(meshes[3])) \
                * dolfin.dot(G[0][0][ext],var_F) * ds(1) \
                + dolfin.assemble(Fs[1] * G[1][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * G[1][2][ext] * dolfin.dx(meshes[2])) \
                 * dolfin.assemble(Fs[3] * G[1][3][ext] * dolfin.dx(meshes[3])) \
                 * dolfin.dot(G[1][0][ext],var_F) * ds(2)

        if nE > 0:
            for old in range(nE):
                l += -dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][0] * PGD_func[3][old] * dolfin.dx(meshes[3])) \
                    * dolfin.inner(param["Cmatrix"][0]*eps(PGD_func[0][old]),eps(var_F)) * dolfin.dx(meshes[0]) \
                    - dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][1] * PGD_func[3][old] * dolfin.dx(meshes[3])) \
                    * dolfin.inner(param["Cmatrix"][1]*eps(PGD_func[0][old]),eps(var_F)) * dolfin.dx(meshes[0])

    if typ == 's':
        l = 0
        for ext in range(len(G[0][1])):
            l += dolfin.assemble(dolfin.dot(G[0][0][ext],Fs[0])*ds(1)) \
                 * dolfin.assemble(Fs[2] * G[0][2][ext] * dolfin.dx(meshes[2])) \
                 * dolfin.assemble(Fs[3] * G[0][3][ext] * dolfin.dx(meshes[3])) \
                 * var_F * G[0][1][ext] * dolfin.dx(meshes[1]) \
                + dolfin.assemble(dolfin.dot(G[1][0][ext],Fs[0])*ds(2)) \
                 * dolfin.assemble(Fs[2] * G[1][2][ext] * dolfin.dx(meshes[2])) \
                 * dolfin.assemble(Fs[3] * G[1][3][ext] * dolfin.dx(meshes[3])) \
                 * var_F * G[1][1][ext] * dolfin.dx(meshes[1])
        if nE > 0:
            for old in range(nE):
                l += -dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][0] * PGD_func[3][old] * dolfin.dx(meshes[3])) \
                    * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                    - dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][1] * PGD_func[3][old] * dolfin.dx(meshes[3])) \
                    * var_F * PGD_func[1][old] * dolfin.dx(meshes[1])

    if typ == 't':
        l = 0
        for ext in range(len(G[0][1])):
            l += dolfin.assemble(dolfin.dot(G[0][0][ext],Fs[0])*ds(1)) \
                 * dolfin.assemble(Fs[1] * G[0][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[3] * G[0][3][ext] * dolfin.dx(meshes[3])) \
                 * var_F * G[0][2][ext] * dolfin.dx(meshes[2]) \
                + dolfin.assemble(dolfin.dot(G[1][0][ext],Fs[0])*ds(2)) \
                 * dolfin.assemble(Fs[1] * G[1][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[3] * G[1][3][ext] * dolfin.dx(meshes[3])) \
                 * var_F * G[1][2][ext] * dolfin.dx(meshes[2])
        if nE > 0:
            for old in range(nE):
                l += -dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][0] * PGD_func[3][old] * dolfin.dx(meshes[3])) \
                    * var_F * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                    - dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[3] * param["Nu_func"][1] * PGD_func[3][old] * dolfin.dx(meshes[3])) \
                    * var_F * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])

    if typ == 'v':
        l = 0
        for ext in range(len(G[0][1])):
            l += dolfin.assemble(dolfin.dot(G[0][0][ext],Fs[0])*ds(1)) \
                 * dolfin.assemble(Fs[1] * G[0][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * G[0][2][ext] * dolfin.dx(meshes[2])) \
                 * var_F * G[0][3][ext] * dolfin.dx(meshes[3]) \
                + dolfin.assemble(dolfin.dot(G[1][0][ext],Fs[0])*ds(2)) \
                 * dolfin.assemble(Fs[1] * G[1][1][ext] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * G[1][2][ext] * dolfin.dx(meshes[2])) \
                 * var_F * G[1][3][ext] * dolfin.dx(meshes[3])
        if nE > 0:
            for old in range(nE):
                l += -dolfin.assemble(dolfin.inner(param["Cmatrix"][0]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * var_F * param["Nu_func"][0] * PGD_func[3][old] * dolfin.dx(meshes[3]) \
                    - dolfin.assemble(dolfin.inner(param["Cmatrix"][1]*eps(PGD_func[0][old]),eps(Fs[0])) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * param["E_func"] * param["E_0"] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                    * var_F * param["Nu_func"][1] * PGD_func[3][old] * dolfin.dx(meshes[3])

    return l

def main_normal(vs, writeFlag=False, name='PGDsolution'):
    '''
        computation of PGD solution for given problem normal
        :param vs: list of function spaces len = num_pgd_var
        :param writeFlag: save files or not
        :return: PGDModel and PGDProblem1
    '''

    PGD_nmax = 10  # PGD enrichment number (max number of PGD modes per coordinate)
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

    # param = {"E_0": 30000, "gamma":25e-6, "E_func": efunc, "Nu_func": [nu_func1, nu_func2],
    #          "Cmatrix": [c1, c2]}
    param = {"E_0": 1, "gamma": 25e-6, "E_func": efunc, "Nu_func": [nu_func1, nu_func2],
             "Cmatrix": [c1, c2]}

    # define separated load expressions
    g1_1 = [dolfin.Constant((0., -0.5))]
    g1_2 = [dolfin.Expression('x[0]', degree=4)]
    g1_3 = [dolfin.Expression('1.0', degree=4)]
    g1_4 = [dolfin.Expression('1.0', degree=4)]

    g2_1 = [dolfin.Constant((0., -1.5))]
    g2_2 = [dolfin.Expression('x[0]', degree=4)]
    g2_3 = [dolfin.Expression('1.0', degree=4)]
    g2_4 = [dolfin.Expression('1.0', degree=4)]

    load = [[g1_1, g1_2, g1_3, g1_4], [g2_1, g2_2, g2_3, g2_4]]

    # PGDProblem class
    pgd_prob = PGDProblem1(name='PGD_xpEv', name_coord=['X', 'P', 'E', 'nu'],
                           modes_info=['U', 'Node', 'Vector'],
                           Vs=vs, bc_fct=create_bc, load=load,
                           param=param, dom_fct=create_dom,
                           rhs_fct=problem_assemble_rhs, lhs_fct=problem_assemble_lhs,
                           probs=prob, seq_fp=seq_fp, PGD_nmax=PGD_nmax)

    # solve displacement problem
    pgd_prob.solve_PGD()  # solve normal

    print('computed:', pgd_prob.name, pgd_prob.amplitude)
    print(pgd_prob.simulation_info)

    pgd_solution = pgd_prob.return_PGD()

    pgd_solution.print_info()

    # save for postprocessing!!
    if writeFlag:
        folder = os.path.join(os.path.dirname(__file__), '..', 'results', name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        f = open(os.path.join(folder, 'git_version_sha.txt'), 'w')
        f.write('used git commit: ' + str(get_git_revision2()))
        f.close()
        pgd_solution.write_hdf5(folder)
        pgd_solution.write_pxdmf(folder)

    return pgd_prob, pgd_solution



'''main program'''
if __name__ == '__main__':

    import logging


    # parameters same for all!!
    ORD = 2
    ORDS = [ORD, ORD, ORD, ORD]
    # ranges extra coordinates
    RANGES = [[0., 2.],  # pmin,pmax
              [0.5, 1.5],  # Emin,Emax
              [0.1, 0.3]]  # numin,numax


    # solve normal PGD problem
    # mesh in x space
    _, v_x = create_meshX([200, 20], ORDS[0])
    # mesh in extra coordinates
    NUM_ELEM = [2, 50, 50]  # p, E, nu
    print('ff', ORDS, ORDS[1:4])
    _, v_e = create_meshesExtra(NUM_ELEM,ORDS[1:4],RANGES)
    # solve
    pgd_prob_normal, pgd_solution_normal = \
        main_normal([v_x] + v_e, writeFlag=False, name='PGDsolution_normal_O%i' % ORD)


