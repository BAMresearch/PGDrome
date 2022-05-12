r'''
    1D PGD example
    linear elastic uniaxial truss with constant load:

      ->->->->->n(x)->->->->->->
    >-------------EA------------<
                L=1
    with E = Emodul factor E_0 and n(x)=loadfactor n_0

    DGL: \int var_eps E eps A dx = \int var_u n dx

    three PGD variables: space, load factor and Emodul factor

    solving PGD problem in standard way for u(x,loadfactor,Emodulfactor)

    returning PGD instance

'''

import unittest
import dolfin
import os

from pgdrome.solver import PGDProblem1
from pgdrome.model import PGDErrorComputation

def create_meshes(num_elem, ord, ranges):
    '''
    :param num_elem: list for each PG CO
    :param ord: list for each PG CO
    :param ranges: list for each PG CO
    :return: meshes and V
    '''

    print('create meshes')

    meshes = list()
    Vs = list()

    dim = len(num_elem)

    for i in range(dim):
        mesh_tmp = dolfin.IntervalMesh(num_elem[i], ranges[i][0], ranges[i][1])
        Vs_tmp = dolfin.FunctionSpace(mesh_tmp, 'P', ord[i])

        meshes.append(mesh_tmp)
        Vs.append(Vs_tmp)

    return meshes, Vs

def create_bc(Vs,dom,param):
    # boundary conditions list

    # only for x problem
    def left(x, on_boundary):
        return x < 0.0 + 1E-5
    def right(x, on_boundary):
        return x > 1.0 - 1E-5

    bc_l = dolfin.DirichletBC(Vs[0], 0., left)
    bc_r = dolfin.DirichletBC(Vs[0], 0., right)
    bcs = [bc_l, bc_r]

    return [bcs, 0, 0]

def problem_assemble_lhs(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem

    if typ == 'r':
        a = dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * param["Efunc"] * Fs[2] * dolfin.dx(meshes[2]))) \
            * var_F.dx(0) * param["E_0"] * fct_F.dx(0) * param["A"] * dolfin.dx(meshes[0])
    if typ == 's':
        a = dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * param["E_0"] * Fs[0].dx(0) * param["A"] * dolfin.dx(meshes[0])) \
             * dolfin.assemble(Fs[2] * param["Efunc"] * Fs[2] * dolfin.dx(meshes[2]))) \
             * var_F * fct_F * dolfin.dx(meshes[1])
    if typ == 't':
        a = dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * param["E_0"] * Fs[0].dx(0) * param["A"] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] *dolfin.dx(meshes[1]))) \
            * var_F * param["Efunc"] * fct_F *dolfin.dx(meshes[2])
    return a

def problem_assemble_rhs(fct_F,var_F,Fs,meshes,dom,param,G,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem
    if typ == 'r':
        l = dolfin.Constant(dolfin.assemble(Fs[1] * G[1][0] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * G[2][0] * dolfin.dx(meshes[2])) )\
            * var_F * G[0][0] * param["A"] * dolfin.dx(meshes[0])
        if nE > 0:
            for old in range(nE):
                l += - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dolfin.dx(meshes[2])))\
                     * var_F.dx(0) * param["E_0"] * PGD_func[0][old].dx(0) * param["A"] *dolfin.dx(meshes[0])

    if typ == 's':
        l = dolfin.Constant(dolfin.assemble(Fs[0] * G[0][0] * param["A"] *dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * G[2][0] * dolfin.dx(meshes[2])) )\
            * var_F * G[1][0] * dolfin.dx(meshes[1])
        if nE > 0:
            for old in range(nE):
                l += - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * param["E_0"] * PGD_func[0][old].dx(0) * param["A"] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                     * var_F * PGD_func[1][old] * dolfin.dx(meshes[1])

    if typ == 't':
        l = dolfin.Constant(dolfin.assemble(Fs[0] * G[0][0] * param["A"] *dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * G[1][0] * dolfin.dx(meshes[1]))) \
            * var_F * G[2][0] * dolfin.dx(meshes[2])
        if nE > 0:
            for old in range(nE):
                l += - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * param["E_0"] * PGD_func[0][old].dx(0) * param["A"] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) )\
                     * var_F * param["Efunc"] * PGD_func[2][old] * dolfin.dx(meshes[2])
    return l

def main(vs, writeFlag=False, name=None):
    '''computation of PGD solution for given problem '''

    # define some parameters
    param = {"A": 1.0, "p_0": 1.0, "E_0": 1.0, "Efunc": dolfin.Expression('x[0]', degree=4)}

    # define separated load expressions
    g1 = [dolfin.Expression('1.0', degree=4)]
    g2 = [dolfin.Expression('p*A*x[0]', p=param['p_0'], A=param['A'], degree=4)]
    g3 = [dolfin.Expression('1.0', degree=4)]

    prob = ['r', 's', 't'] # problems according problem_assemble_fcts
    seq_fp = [0, 1, 2]  # default sequence of Fixed Point iteration
    PGD_nmax = 10       # max number of PGD modes

    pgd_prob = PGDProblem1(name='Uniaxial1D-PGD-XPE', name_coord=['X', 'P', 'E'],
                           modes_info=['U_x', 'Node', 'Scalar'],
                           Vs=vs, dom=0, bc_fct=create_bc, load=[g1,g2,g3],
                           param=param, rhs_fct=problem_assemble_rhs,
                           lhs_fct=problem_assemble_lhs, probs=prob, seq_fp=seq_fp,
                           PGD_nmax=PGD_nmax)
    #
    # possible solver parameters (if not given then default values will be used!)
    # pgd_prob.max_fp_it = 5
    # pgd_prob.stop_fp = 'norm' #'delta'
    # pgd_prob.tol_fp_it = 1e-5
    # pgd_prob.tol_abs = 1e-4
    pgd_prob.solve_PGD(_problem='linear') # solve

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance
    # pgd_s.print_info()
    print(pgd_prob.simulation_info)
    print('Amplitude:', pgd_prob.amplitude)

    # save for postprocessing!!
    if writeFlag:
        folder = os.path.join(os.path.dirname(__file__), '..', 'results', name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        f = open(os.path.join(folder, 'git_version_sha.txt'), 'w')
        # f.write('used git commit: ' + str(get_git_revision2()))
        f.close()
        pgd_s.write_hdf5(folder)
        pgd_s.write_pxdmf(folder)

    return pgd_s

class PGDproblem(unittest.TestCase):

    def setUp(self):
        # global parameters
        self.ord = 2  # 1 # 2 # order for each mesh
        self.ords = [self.ord, self.ord, self.ord]
        self.ranges = [[0., 1.],  # xmin, xmax
                  [-1., 3.],  # pmin,pmax
                  [0.2, 2.0]]  # Emin,Emax

        self.write = False # set to True to save pxdmf file

        self.p = 2.0
        self.E = 1.5
        self.x = 0.5

        self.analytic_solution = 1.0*self.p / (2 * 1.0*self.E * 1.0) * (-self.x * self.x + 1.0 * self.x)

    def TearDown(self):
        pass

    def test_standard_solver(self):
        # define meshes
        meshes, vs = create_meshes([113, 2, 100], self.ords, self.ranges)  # start meshes
        # solve PGD problem
        pgd_test = main(vs, writeFlag=self.write, name='PGDsolution_O%i' % self.ord)
        
        # Sampling
        error_uPGD = PGDErrorComputation()
        min_bnd = [self.ranges[0][0], self.ranges[1][0], self.ranges[2][0]] # Minimum boundary
        max_bnd = [self.ranges[0][1], self.ranges[1][1], self.ranges[2][1]] # Maximum boundary

        data_test = error_uPGD.sampling_LHS(10, len(self.ranges), min_bnd, max_bnd) # Sampling
        
        # evaluate
        errorL2 = np.zeros(len(data_test))
        for i in range(len(data_test)):
            
            # Solve PGD:
            #-----------------------------------
            u_pgd = pgd_test.evaluate(0, [1, 2], [data_test[i][1], data_test[i][2]], 0)
            
            # Analytical solution
            #-----------------------------------
            u_anl = 1.0*data_test[i][1] / (2 * 1.0*data_test[i][2] * 1.0) * (-data_test[i][0] * data_test[i][0] + 1.0 * data_test[i][0])
            
            # Compute error
            #-----------------------------------
            error_uPGD = PGDErrorComputation()
            errorL2[i] = error_uPGD.compute_SampleError(u_anl, u_pgd(data_test[i][0]))
        
        mean_errorL2 = np.mean(errorL2)
        max_errorL2 = np.max(errorL2)
        print('Mean error',mean_errorL2)
        print('Max. error',max_errorL2)
        
        self.assertTrue(mean_errorL2<0.01)


if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=logging.DEBUG)

    unittest.main()
