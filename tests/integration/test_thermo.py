'''
    simple 1D PGD example (heat equation with a point heat source) with three PGD variables (space, time and heat input)

    solving PGD problem in standard way using FEM

    returning PGDModel (as forward model) or PGD instance

'''

import unittest
import dolfin
import os
import numpy as np

from pgdrome.solver import PGDProblem1

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
        Vs_tmp = dolfin.FunctionSpace(mesh_tmp, 'CG', ord[i])

        meshes.append(mesh_tmp)
        Vs.append(Vs_tmp)

    return meshes, Vs

def create_bc(Vs,dom,param):
    # boundary conditions list

    # Initial condition
    def init(x, on_boundary):
        return x < 0.0 + 1E-5

    initCond = dolfin.DirichletBC(Vs[1], 0, init)

    return [0, initCond, 0]

def problem_assemble_lhs(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem

    if typ == 'r':
        a = dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * fct_F * var_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * fct_F.dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
    if typ == 's':
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * fct_F.dx(0) * var_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * fct_F * var_F * dolfin.dx(meshes[1])
    if typ == 'w':
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1].dx(0) * dolfin.dx(meshes[1]))) \
            * param["rho"] * param["c_p"] * fct_F * var_F * dolfin.dx(meshes[2])\
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))) \
            * param["k"] * fct_F * var_F * dolfin.dx(meshes[2])
    return a

def problem_assemble_rhs(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem

    IC_r = param["IC_r"]
    IC_s = param["IC_s"]
    IC_Eta = param["IC_Eta"]

    if typ == 'r':
        l = dolfin.Constant(dolfin.assemble(Q[1][0] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Q[2][0] * Fs[2] * dolfin.dx(meshes[2]))) \
            * Q[0][0] * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(dolfin.assemble(IC_s.dx(0) * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * IC_r * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(dolfin.assemble(IC_s * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * IC_r.dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
        if nE > 0:
            for old in range(nE):
                l +=- dolfin.Constant(dolfin.assemble(PGD_func[1][old].dx(0) * Fs[1] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["c_p"] * PGD_func[0][old] * var_F * dolfin.dx(meshes[0]) \
                    - dolfin.Constant(dolfin.assemble(PGD_func[1][old] * Fs[1] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["k"] * PGD_func[0][old].dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
    if typ == 's':
        l = dolfin.Constant(dolfin.assemble(Q[0][0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Q[2][0] * Fs[2] * dolfin.dx(meshes[2]))) \
            * Q[1][0] * var_F * dolfin.dx(meshes[1]) \
            - dolfin.Constant(dolfin.assemble(IC_r * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * IC_s.dx(0) * var_F * dolfin.dx(meshes[1]) \
            - dolfin.Constant(dolfin.assemble(IC_r.dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * IC_s * var_F * dolfin.dx(meshes[1])
        if nE > 0:
            for old in range(nE):
                l +=- dolfin.Constant(dolfin.assemble(PGD_func[0][old] * Fs[0] * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["c_p"] * PGD_func[1][old].dx(0) * var_F * dolfin.dx(meshes[1]) \
                    - dolfin.Constant(dolfin.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["k"] * PGD_func[1][old] * var_F * dolfin.dx(meshes[1])
    if typ == 'w':
        l = dolfin.Constant(dolfin.assemble(Q[0][0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Q[1][0] * Fs[1] * dolfin.dx(meshes[1]))) \
            * Q[2][0] * var_F * dolfin.dx(meshes[2]) \
            - dolfin.Constant(dolfin.assemble(IC_r * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC_s.dx(0) * Fs[1] * dolfin.dx(meshes[1]))) \
            * param["rho"] * param["c_p"] * IC_Eta * var_F * dolfin.dx(meshes[2]) \
            - dolfin.Constant(dolfin.assemble(IC_r.dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC_s * Fs[1] * dolfin.dx(meshes[1]))) \
            * param["k"] * IC_Eta * var_F * dolfin.dx(meshes[2])
        if nE > 0:
            for old in range(nE):
                l +=- dolfin.Constant(dolfin.assemble(PGD_func[0][old] * Fs[0] * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[1][old].dx(0) * Fs[1] * dolfin.dx(meshes[1]))) \
                    * param["rho"] * param["c_p"] * PGD_func[2][old] * var_F * dolfin.dx(meshes[2]) \
                    - dolfin.Constant(dolfin.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[1][old] * Fs[1] * dolfin.dx(meshes[1]))) \
                    * param["k"] * PGD_func[2][old] * var_F * dolfin.dx(meshes[2])
    return l

def main(vs, writeFlag=False, name=None):
    '''computation of PGD solution for given problem '''

    # define some parameters
    param = {"rho": 7100, "c_p": 3100, "k": 100}

    # define nonhomogeneous dirichlet IC in time s
    param.update({'IC_r': dolfin.interpolate(dolfin.Expression('1.0', degree=4),vs[0])})
    param.update({'IC_s': dolfin.interpolate(dolfin.Expression('(x[0] < 0.0 + 1E-8) ? 25 : 0', degree=4),vs[1])})
    param.update({'IC_Eta': dolfin.interpolate(dolfin.Expression('1.0', degree=4),vs[2])})

    # define heat source in x, t and eta
    q1 = [dolfin.Expression('6*sqrt(3)*P / ((af+ar)*af*af*pow(pi,3/2)) * exp(-3*(pow(x[0]-xc,2)/pow(af,2)))', degree=4, P=2500, af=0.002, ar=0.002, xc=0.05)]
    q2 = [dolfin.interpolate(dolfin.Expression('1.0', degree=4),vs[1])]
    q3 = [dolfin.interpolate(dolfin.Expression('x[0]', degree=4), vs[2])]

    prob = ['r', 's', 'w'] # problems according problem_assemble_fcts
    seq_fp = np.arange(len(vs))  # default sequence of Fixed Point iteration
    PGD_nmax = 10       # max number of PGD modes

    pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-XTEta', name_coord=['X', 'T', 'Eta'],
                           modes_info=['T_x', 'Node', 'Scalar'],
                           Vs=vs, dom=0, bc_fct=create_bc, load=[q1,q2,q3],
                           param=param, rhs_fct=problem_assemble_rhs,
                           lhs_fct=problem_assemble_lhs, probs=prob, seq_fp=seq_fp,
                           PGD_nmax=PGD_nmax)

    pgd_prob.max_fp_it = 50
    pgd_prob.stop_fp = 'norm'
    pgd_prob.tol_fp_it = 1e-5
    pgd_prob.tol_abs = 1e-4
    pgd_prob.solve_PGD(_problem='linear') # solve normal
    print(pgd_prob.simulation_info)

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance
    # pgd_s.print_info()

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
        self.ord = 1  # 1 # 2 # order for each mesh
        self.ords = [self.ord, self.ord, self.ord]
        self.ranges = [[0., 0.1],  # xmin, xmax
                  [0., 10.],  # tmin, tmax
                  [0.7, 0.9]]  # etamin, etamax

        self.write = False # set to True to save pxdmf file

        self.x = 0.01545
        self.t = 10
        self.eta = 0.8

    def TearDown(self):
        pass

    def test_standard_solver(self):
        # define meshes
        meshes, vs = create_meshes([400, 100, 10], self.ords, self.ranges)  # start meshes

        # solve PGD problem
        pgd_test = main(vs, writeFlag=self.write, name='PGDsolution_O%i' % self.ord)

        # evaluate
        # Merke für t=0 gibt dies nur den homogenen Teil, nicht den inhomogenen
        u_pgd = pgd_test.evaluate(0, [1, 2], [self.t, self.eta], 0)

        # import FEM solution
        file_fem=os.path.join(os.path.dirname(__file__),"test_thermo_data.dat")
        time, compareValues = np.loadtxt(file_fem,unpack=True)
        for i in range(time.__len__()):
            if dolfin.near(time[i],self.t,eps=1E-8):
                compareValue = compareValues[i]
                break

        # compare solutions
        print('evaluate PGD', u_pgd(self.x), 'ref solution', compareValue)
        print('difference: ', np.abs(u_pgd(self.x)-compareValue))
        self.assertAlmostEqual(u_pgd(self.x), compareValue, places=1)

if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.DEBUG)

    unittest.main()
