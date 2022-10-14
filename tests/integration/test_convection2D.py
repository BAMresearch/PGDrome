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

    return meshes, Vs #x,y,t,h

def create_bc(Vs,dom,param):
    # boundary conditions list

    # Initial condition
    def init(x, on_boundary):
        return x < 0.0 + 1E-5

    initCond = dolfin.DirichletBC(Vs[2], 0, init)

    return [0, 0, initCond, 0] #x,y,t,h


def create_doms_pgd(Vs,param):
    boundary_x = dolfin.MeshFunction("size_t", Vs[0].mesh(), Vs[0].mesh().topology().dim() - 1)
    boundary_x.set_all(0)

    boundary_y = dolfin.MeshFunction("size_t", Vs[1].mesh(), Vs[1].mesh().topology().dim() - 1)
    boundary_y.set_all(0)

    class zero(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], 0., dolfin.DOLFIN_EPS)

    class ly(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], param['ly']/param['L_0'], dolfin.DOLFIN_EPS)

    class lx(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], param['lx']/param['L_0'], dolfin.DOLFIN_EPS)

    Zero_x = zero()
    Zero_x.mark(boundary_x, 1)
    LX = lx()
    LX.mark(boundary_x, 2)

    Zero_y = zero()
    Zero_y.mark(boundary_y, 1)
    LY = ly()
    LY.mark(boundary_y, 2)

    return [boundary_x,boundary_y,0,0] # x,y,t,h

def problem_assemble_lhs_FDtime(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem
    ds_x = dolfin.Measure('ds', domain=meshes[0], subdomain_data=dom[0])
    ds_y = dolfin.Measure('ds', domain=meshes[1], subdomain_data=dom[1])
    ds = [ds_x, ds_y]

    # factors needed for all/most
    alpha_x1 = dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
    alpha_x2 = dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0]))
    alpha_y1 = dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
    alpha_y2 = dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1]))
    alpha_t1 = Fs[2].vector()[:].transpose() @ param['D1_up_t'] @ Fs[2].vector()[:]
    alpha_t2 = Fs[2].vector()[:].transpose() @ param['M_t'] @ Fs[2].vector()[:]
    alpha_h1 = dolfin.assemble(Fs[3] * Fs[3] * dolfin.dx(meshes[3]))
    alpha_h2 = dolfin.assemble(Fs[3] * param['h_fct'] * Fs[3] * dolfin.dx(meshes[3]))

    if typ == 'r': #x
        a = dolfin.Constant(alpha_y1 * alpha_t1 * alpha_h1) * param['a1'] * param["rho"] * param["cp"] \
            * fct_F * var_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(alpha_y1 * alpha_t2 * alpha_h1) * param['a2'] * param["k"] \
            * fct_F.dx(0) * var_F.dx(0) * dolfin.dx(meshes[0]) \
            + dolfin.Constant(alpha_y2 * alpha_t2 * alpha_h1) * param['a2'] * param["k"]\
            * fct_F * var_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(alpha_y1 * alpha_t2 * alpha_h2) * param['h1']\
            * fct_F * var_F * ds[0](2) \
            + dolfin.Constant(dolfin.assemble(Fs[1]*Fs[1]*ds[1](2)) * alpha_t2 * alpha_h2) * param['h1']\
            * fct_F * var_F * dolfin.dx(meshes[0])

    if typ == 's': #y
        a = dolfin.Constant(alpha_x1 * alpha_t1 * alpha_h1) * param['a1'] * param["rho"] * param["cp"] \
            * fct_F * var_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(alpha_x2 * alpha_t2 * alpha_h1) * param['a2'] * param["k"]\
            * fct_F * var_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(alpha_x1 * alpha_t2  * alpha_h1) * param['a2'] * param["k"]\
            * fct_F.dx(0) * var_F.dx(0) * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0]*Fs[0]*ds[0](2)) * alpha_t2 * alpha_h2) * param['h1']\
            * fct_F * var_F * dolfin.dx(meshes[1])\
            + dolfin.Constant(alpha_x1 * alpha_t2 * alpha_h2) * param['h1']\
            * fct_F * var_F * ds[1](2)

    if typ == 'v': #t
        a = alpha_x1 * alpha_y1 * alpha_h1 * param['a1'] * param["rho"] * param["cp"] * param['D1_up_t'] \
            + alpha_x2 * alpha_y1 * alpha_h1 * param['a2'] * param["k"] * param['M_t'] \
            + alpha_x1 * alpha_y2 * alpha_h1 * param['a2'] * param["k"] * param['M_t'] \
            + dolfin.assemble(Fs[0]*Fs[0]*ds[0](2)) * alpha_y1 * alpha_h2 * param['h1'] * param['M_t'] \
            + alpha_x1 * dolfin.assemble(Fs[1] * Fs[1] * ds[1](2)) * alpha_h2 * param['h1'] * param['M_t']
            # add initial condition
        a[:, param['bc_idx']] = 0
        a[param['bc_idx'], :] = 0
        a[param['bc_idx'], param['bc_idx']] = 1

    if typ == 'w': #h
        a = dolfin.Constant(alpha_x1 * alpha_y1 * alpha_t1) * param['a1'] * param["rho"] * param["cp"]\
            * fct_F * var_F * dolfin.dx(meshes[3]) \
            + dolfin.Constant(alpha_x2 * alpha_y1 * alpha_t2) * param['a2'] * param["k"]\
            * fct_F * var_F * dolfin.dx(meshes[3]) \
            + dolfin.Constant(alpha_x1 * alpha_y2 * alpha_t2) * param['a2'] * param["k"]\
            * fct_F * var_F * dolfin.dx(meshes[3]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * ds[0](2)) * alpha_y1 * alpha_t2) * param['h1']\
            * fct_F * param['h_fct'] * var_F * dolfin.dx(meshes[3]) \
            + dolfin.Constant(alpha_x1 * dolfin.assemble(Fs[1] * Fs[1] * ds[1](2)) * alpha_t2) * param['h1']\
            * fct_F * param['h_fct'] * var_F * dolfin.dx(meshes[3])

    return a

def problem_assemble_rhs_FDtime(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem

    IC = [param["IC_x"], param["IC_y"], param["IC_t"], param["IC_h"]]
    Tamb = [param['Tamb_x'], param['Tamb_y'], param['Tamb_t'], param['Tamb_h']]

    ds_x = dolfin.Measure('ds', domain=meshes[0], subdomain_data=dom[0])
    ds_y = dolfin.Measure('ds', domain=meshes[1], subdomain_data=dom[1])
    ds = [ds_x, ds_y]

    # print('----CHECK BOUNDARY----')
    # print('A1', dolfin.assemble(1 * ds[1](2)) * dolfin.assemble(1 * dolfin.dx(meshes[0]))) #dx*dy
    # print('A2', dolfin.assemble(1 * ds[0](2)) * dolfin.assemble(1 * dolfin.dx(meshes[1]))) #dx*dy
    # print('full A', dolfin.assemble(1 * dolfin.dx(meshes[1])) * dolfin.assemble(1 * dolfin.dx(meshes[0])))
    # summarize IC and PGD_func[old]
    Told = []
    for dd in range(len(IC)):
        Ti=[IC[dd]]
        if nE > 0:
            for old in range(nE):
                Ti.append(PGD_func[dd][old])
        # print('check', len(Ti))
        Told.append(Ti)
    # print('check len Told', len(Told), len(Told[0]))

    # factors Tamb
    beta_x1 = dolfin.assemble(Tamb[0] * Fs[0] * dolfin.dx(meshes[0]))
    beta_y1 = dolfin.assemble(Tamb[1] * Fs[1] * dolfin.dx(meshes[1]))
    beta_t2 = Fs[2].vector()[:].transpose() @ param['M_t'] @ Tamb[2].vector()[:]
    beta_h2 = dolfin.assemble(Tamb[3] * param['h_fct'] * Fs[3] * dolfin.dx(meshes[3]))

    if typ == 'r': #x
        l = dolfin.Constant(beta_y1 * beta_t2 * beta_h2) * param['h1'] \
            * Tamb[0] * var_F * ds[0](2) \
            + dolfin.Constant(dolfin.assemble(Tamb[1] * Fs[1] * ds[1](2)) * beta_t2 * beta_h2) * param['h1'] \
            * Tamb[0] * var_F * dolfin.dx(meshes[0])
        for old in range(len(Told[0])):
            alpha_old_y1 = dolfin.assemble(Told[1][old] * Fs[1] * dolfin.dx(meshes[1]))
            alpha_old_y2 = dolfin.assemble(Told[1][old].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1]))
            alpha_old_t1 = Fs[2].vector()[:].transpose() @ param['D1_up_t'] @ Told[2][old].vector()[:]
            alpha_old_t2 = Fs[2].vector()[:].transpose() @ param['M_t'] @ Told[2][old].vector()[:]
            alpha_old_h1 = dolfin.assemble(Told[3][old] * Fs[3] * dolfin.dx(meshes[3]))
            alpha_old_h2 = dolfin.assemble(Told[3][old] * param['h_fct'] * Fs[3] * dolfin.dx(meshes[3]))

            l += - dolfin.Constant(alpha_old_y1 * alpha_old_t1 * alpha_old_h1) * param['a1'] * param["rho"] * param["cp"] \
                * Told[0][old] * var_F * dolfin.dx(meshes[0]) \
                - dolfin.Constant(alpha_old_y1 * alpha_old_t2 * alpha_old_h1) * param['a2'] * param["k"] \
                * Told[0][old].dx(0) * var_F.dx(0) * dolfin.dx(meshes[0]) \
                - dolfin.Constant(alpha_old_y2 * alpha_old_t2 * alpha_old_h1) * param['a2'] * param["k"] \
                * Told[0][old] * var_F * dolfin.dx(meshes[0]) \
                - dolfin.Constant(alpha_old_y1 * alpha_old_t2 * alpha_old_h2) * param['h1'] \
                * Told[0][old] * var_F * ds[0](2) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * ds[1](2)) * alpha_old_t2 * alpha_old_h2) * param['h1'] \
                * Told[0][old] * var_F * dolfin.dx(meshes[0])

    if typ == 's': #y
        l = dolfin.Constant(dolfin.assemble(Tamb[0] * Fs[0] * ds[0](2)) * beta_t2 * beta_h2) * param['h1']\
            * Tamb[1] * var_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(beta_x1 * beta_t2 * beta_h2) * param['h1']\
            * Tamb[1] * var_F * ds[1](2)
        for old in range(len(Told[0])):
            alpha_old_x1 = dolfin.assemble(Told[0][old] * Fs[0] * dolfin.dx(meshes[0]))
            alpha_old_x2 = dolfin.assemble(Told[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0]))
            alpha_old_t1 = Fs[2].vector()[:].transpose() @ param['D1_up_t'] @ Told[2][old].vector()[:]
            alpha_old_t2 = Fs[2].vector()[:].transpose() @ param['M_t'] @ Told[2][old].vector()[:]
            alpha_old_h1 = dolfin.assemble(Told[3][old] * Fs[3] * dolfin.dx(meshes[3]))
            alpha_old_h2 = dolfin.assemble(Told[3][old] * param['h_fct'] * Fs[3] * dolfin.dx(meshes[3]))
            l += -dolfin.Constant(alpha_old_x1 * alpha_old_t1 * alpha_old_h1) * param['a1'] * param["rho"] * param["cp"] \
                * Told[1][old] * var_F * dolfin.dx(meshes[1]) \
                - dolfin.Constant(alpha_old_x2 * alpha_old_t2 * alpha_old_h1) * param['a2'] * param["k"]\
                * Told[1][old] * var_F * dolfin.dx(meshes[1]) \
                - dolfin.Constant(alpha_old_x1 * alpha_old_t2  * alpha_old_h1) * param['a2'] * param["k"]\
                * Told[1][old].dx(0) * var_F.dx(0) * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0]*Fs[0]*ds[0](2)) * alpha_old_t2 * alpha_old_h2) * param['h1']\
                * Told[1][old] * var_F * dolfin.dx(meshes[1])\
                - dolfin.Constant(alpha_old_x1 * alpha_old_t2 * alpha_old_h2) * param['h1']\
                * Told[1][old] * var_F * ds[1](2)

    if typ == 'v': #t
        l = dolfin.assemble(Tamb[0] * Fs[0] * ds[0](2)) * beta_y1 * beta_h2 * param['h1'] * param['M_t'] @ Tamb[2].vector()[:] \
            + beta_x1 * dolfin.assemble(Tamb[1] * Fs[1] * ds[1](2)) * beta_h2 * param['h1'] * param['M_t'] @ Tamb[2].vector()[:]

        for old in range(len(Told[0])):
            alpha_old_x1 = dolfin.assemble(Told[0][old] * Fs[0] * dolfin.dx(meshes[0]))
            alpha_old_x2 = dolfin.assemble(Told[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0]))
            alpha_old_y1 = dolfin.assemble(Told[1][old] * Fs[1] * dolfin.dx(meshes[1]))
            alpha_old_y2 = dolfin.assemble(Told[1][old].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1]))
            alpha_old_h1 = dolfin.assemble(Told[3][old] * Fs[3] * dolfin.dx(meshes[3]))
            alpha_old_h2 = dolfin.assemble(Told[3][old] * param['h_fct'] * Fs[3] * dolfin.dx(meshes[3]))

            l += - alpha_old_x1 * alpha_old_y1 * alpha_old_h1 * param['a1'] * param["rho"] * param["cp"]\
                * param['D1_up_t'] @ Told[2][old].vector()[:] \
                - alpha_old_x2 * alpha_old_y1 * alpha_old_h1 * param['a2'] * param["k"]\
                * param['M_t'] @ Told[2][old].vector()[:]\
                - alpha_old_x1 * alpha_old_y2 * alpha_old_h1 * param['a2'] * param["k"]\
                * param['M_t'] @ Told[2][old].vector()[:]\
                - dolfin.assemble(Fs[0] * Fs[0] * ds[0](2)) * alpha_old_y1 * alpha_old_h2 * param['h1']\
                * param['M_t'] @ Told[2][old].vector()[:]\
                - alpha_old_x1 * dolfin.assemble(Fs[1] * Fs[1] * ds[1](2)) * alpha_old_h2 * param['h1']\
                * param['M_t']@ Told[2][old].vector()[:]

        # add initial condition
        l[param['bc_idx']] = 0

    if typ == 'w': #h
        l = dolfin.Constant(dolfin.assemble(Tamb[0] * Fs[0] * ds[0](2)) * beta_y1 * beta_t2) * param['h1']\
            * Tamb[3] * param['h_fct'] * var_F * dolfin.dx(meshes[3]) \
            + dolfin.Constant(beta_x1 * dolfin.assemble(Tamb[1] * Fs[1] * ds[1](2)) * beta_t2) * param['h1']\
            * Tamb[3] * param['h_fct'] * var_F * dolfin.dx(meshes[3])
        for old in range(len(Told[0])):
            alpha_old_x1 = dolfin.assemble(Told[0][old] * Fs[0] * dolfin.dx(meshes[0]))
            alpha_old_x2 = dolfin.assemble(Told[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0]))
            alpha_old_y1 = dolfin.assemble(Told[1][old] * Fs[1] * dolfin.dx(meshes[1]))
            alpha_old_y2 = dolfin.assemble(Told[1][old].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1]))
            alpha_old_t1 = Fs[2].vector()[:].transpose() @ param['D1_up_t'] @ Told[2][old].vector()[:]
            alpha_old_t2 = Fs[2].vector()[:].transpose() @ param['M_t'] @ Told[2][old].vector()[:]
            l += -dolfin.Constant(alpha_old_x1 * alpha_old_y1 * alpha_old_t1) * param['a1'] * param["rho"] * param["cp"]\
                * Told[3][old] * var_F * dolfin.dx(meshes[3]) \
                - dolfin.Constant(alpha_old_x2 * alpha_old_y1 * alpha_old_t2) * param['a2'] * param["k"]\
                * Told[3][old] * var_F * dolfin.dx(meshes[3]) \
                - dolfin.Constant(alpha_old_x1 * alpha_old_y2 * alpha_old_t2) * param['a2'] * param["k"]\
                * Told[3][old] * var_F * dolfin.dx(meshes[3]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * ds[0](2)) * alpha_old_y1 * alpha_old_t2) * param['h1']\
                * Told[3][old] * param['h_fct'] * var_F * dolfin.dx(meshes[3]) \
                - dolfin.Constant(alpha_old_x1 * dolfin.assemble(Fs[1] * Fs[1] * ds[1](2)) * alpha_old_t2) * param['h1']\
                * Told[3][old] * param['h_fct'] * var_F * dolfin.dx(meshes[3])
    return l


def create_PGD(param={}, vs=[]):
    # define nonhomogeneous dirichlet IC
    param.update({'IC_x': dolfin.interpolate(param['IC_x'], vs[0])})
    param.update({'IC_y': dolfin.interpolate(param['IC_y'], vs[1])})
    param.update({'IC_t': dolfin.interpolate(param['IC_t'], vs[2])})
    param.update({'IC_h': dolfin.interpolate(param['IC_h'], vs[3])})

    param.update({'Tamb_x': dolfin.interpolate(dolfin.Expression('1.0', degree=1), vs[0])})
    param.update({'Tamb_y': dolfin.interpolate(dolfin.Expression('1.0', degree=1), vs[1])})
    param.update({'Tamb_t': dolfin.interpolate(param['Tamb_fct'], vs[2])})
    param.update({'Tamb_h': dolfin.interpolate(dolfin.Expression('1.0', degree=1), vs[3])})

    param.update({'h_fct': dolfin.interpolate(param['h_fct'], vs[3])})

    # create FD matrices from meshes
    t_dofs = np.array(vs[2].tabulate_dof_coordinates()[:].flatten())
    t_sort = np.argsort(t_dofs)
    M_t, _, D1_up_t = FD_matrices(t_dofs[t_sort])
    param['M_t'], param['D1_up_t'] = M_t[t_sort, :][:, t_sort], D1_up_t[t_sort, :][:, t_sort]
    param['bc_idx'] = np.where(t_dofs == 0)[0]
    ass_rhs = problem_assemble_rhs_FDtime
    ass_lhs = problem_assemble_lhs_FDtime
    solve_modes = ["FEM", "FEM", "FD", "FEM"]

    pgd_prob = PGDProblem(name='2Dconvection-PGD-XyTH', name_coord=['X', 'Y', 't', 'h'],
                          modes_info=['T', 'Node', 'Scalar'],
                          Vs=vs, dom_fct=create_doms_pgd, bc_fct=create_bc, load=0,
                          param=param, rhs_fct=ass_rhs,
                          lhs_fct=ass_lhs, probs=['r', 's', 'v', 'w'], seq_fp=np.arange(len(vs)),
                          PGD_nmax=50)

    pgd_prob.MM = [0, 0, param['M_t'], 0]  # for norms!

    pgd_prob.stop_fp = 'norm'
    pgd_prob.max_fp_it = 50
    pgd_prob.tol_fp_it = 1e-3
    # pgd_prob.fp_init = 'randomized'
    pgd_prob.norm_modes = 'stiff'
    pgd_prob.PGD_tol = 1e-9  # as stopping criterion

    pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes)

    print(pgd_prob.simulation_info)
    print('PGD Amplitude', pgd_prob.amplitude)

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance

    return pgd_s, param

def create_doms_fem(v,param):
    boundary = dolfin.MeshFunction("size_t", v.mesh(), v.mesh().topology().dim()-1)
    boundary.set_all(0)

    class Bottom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[1], 0., dolfin.DOLFIN_EPS)

    class Top(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[1], param['ly']/param['L_0'], dolfin.DOLFIN_EPS)

    class Left(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], 0., dolfin.DOLFIN_EPS)

    class Right(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], param['lx']/param['L_0'], dolfin.DOLFIN_EPS)

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

        self.F = self.param['a1'] * self.param["rho"] * self.param["cp"] * T * v * dolfin.dx() \
                 + self.param['a2'] * self.dt * self.param["k"] *  dolfin.dot(dolfin.grad(T), dolfin.grad(v)) * dolfin.dx() \
                 + self.dt * self.param['h1'] * self.h * (T - self.T_amb) * v * ds(2) \
                 + self.dt * self.param['h1'] * self.h * (T - self.T_amb) * v * ds(4) \
                 - self.param['a1'] * self.param["rho"] * self.param["cp"] * self.T_n * v * dolfin.dx()
                   # + self.dt * self.param['h1'] * self.h * (T - self.T_amb) * v * ds(1) \
                   # + self.dt * self.param['h1'] * self.h * (T - self.T_amb) * v * ds(3) \ # more convection boundaries

        self.point = point

        # print('----CHECK BOUNDARY ---')
        # print('A2', dolfin.assemble(1 * ds(2)))
        # print('A4', dolfin.assemble(1 * ds(4)))

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
        self.param = {"rho": 7100., "cp": 3100., "k": 100., 'Tamb': 25., 'T_ic': 500.,
                      'lx': 0.15, 'ly': 0.1, 'lt': 20.,
                      'L_0': 1, 't_0': 1., 'T_0': 1.} # possiblity to make the equation dimless
                      #'L_0': 0.1, 't_0': 100., 'T_0': 500. }

        # factors for pde
        self.param['a1'] = self.param['T_0'] / self.param['t_0']
        self.param['a2'] = self.param['T_0'] / self.param['L_0'] ** 2.
        self.param['h1'] = self.param['T_0'] / self.param['L_0']

        self.ranges = [[0., self.param['lx']/ self.param['L_0']],  # xmin, xmax
                       [0., self.param['ly']/ self.param['L_0']],  # ymin, ymax
                       [0., self.param['lt']/ self.param['t_0']],  # tmin, tmax
                       [40., 150.]]  # hmin, hmax

        self.ords = [1, 1, 1, 1]  # x, y, t, h
        self.femorder = 2
        # self.elems = [20, 10, 20, 10]
        self.elems = [80, 50, 20, 100]
        # self.elems = [200, 100, 20, 100]

        self.h_fixed = [50, 100]
        self.point = (self.param['lx'] / self.param['L_0'], self.param['ly']/(2*self.param['L_0']))
        self.fixed_dim = 2 # time

        self.plotting = True
        # self.plotting = False

    def TearDown(self):
        pass

    def test_reference(self):

        self.param['Tamb_fct'] = dolfin.Expression('Tamb', degree=1, Tamb=self.param["Tamb"] / self.param[
            'T_0'])  # ambient condition constant

        # PGD :
        self.param['IC_x'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_y'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_t'] = dolfin.Expression('T', degree=1,
                                               T=self.param["T_ic"] / self.param['T_0'])  # initial condition FEM
        self.param['IC_h'] = dolfin.Expression('1.0', degree=1)

        self.param['h_fct'] = dolfin.Expression('x[0]',degree=1)

        meshes, vs = create_meshes(self.elems, self.ords, self.ranges)
        pgd_fd, param = create_PGD(param=self.param, vs=vs)

        # evaluation at one point with fixed h
        u_pgd_01 = pgd_fd.evaluate(self.fixed_dim, [0,1,3], [self.point[0], self.point[1], self.h_fixed[0]], 0)
        upgd_bc_01 = u_pgd_01.compute_vertex_values()[:] + \
                     param['IC_x'](self.point[0]) * param['IC_y'](self.point[1]) * param['IC_t'].compute_vertex_values()[:] * param["IC_h"](
            self.h_fixed[0])
        print('PGD', np.array(upgd_bc_01) * self.param['T_0'])

        u_pgd_02 = pgd_fd.evaluate(self.fixed_dim, [0, 1, 3], [self.point[0], self.point[1], self.h_fixed[1]], 0)
        upgd_bc_02 = u_pgd_02.compute_vertex_values()[:] + \
                     param['IC_x'](self.point[0]) * param['IC_y'](self.point[1]) * param[
                                                                                       'IC_t'].compute_vertex_values()[
                                                                                   :] * param["IC_h"](
            self.h_fixed[1])
        print('PGD', np.array(upgd_bc_02) * self.param['T_0'])

        # reconstruct x-y field for given h and last time step
        X = np.array(meshes[0].coordinates()[:]).flatten()
        Y = np.array(meshes[1].coordinates()[:]).flatten()
        Z = np.ones((len(Y),len(X)))
        for idx_x, xi in enumerate(X):
            for idx_y, yi in enumerate(Y):
                u_tmp = pgd_fd.evaluate(self.fixed_dim, [0, 1, 3], [xi, yi, self.h_fixed[0]], 0)
                uu_tmp = u_tmp.compute_vertex_values()[:] + \
                          param['IC_x'](xi) * param['IC_y'](yi) \
                          * param['IC_t'].compute_vertex_values()[:] * param["IC_h"](self.h_fixed[0])
                Z[idx_y,idx_x] = self.param['T_0'] * uu_tmp[-1]

        # FEM
        self.param['IC_t'] = dolfin.Expression('T', degree=1, T=self.param["T_ic"]/self.param['T_0'])  # initial condition FEM

        mesh_xy = dolfin.RectangleMesh(dolfin.Point(self.ranges[0][0],self.ranges[1][0]),dolfin.Point(self.ranges[0][1],self.ranges[1][1]), self.elems[0], self.elems[1])
        v_xy = dolfin.FunctionSpace(mesh_xy,'CG',self.femorder)
        v_t = vs[2]
        doms_fem = create_doms_fem(v_xy,self.param)

        u_fem, u_fem2_01 = Reference(param=self.param, vs=[v_xy, v_t], dom = doms_fem, point=self.point)(
            [self.param['lt'], self.h_fixed[0]])
        print('FEM',np.array(u_fem2_01)*self.param['T_0'])

        _, u_fem2_02 = Reference(param=self.param, vs=[v_xy, v_t], dom=doms_fem, point=self.point)(
            [self.param['lt'], self.h_fixed[1]])
        print('FEM', np.array(u_fem2_02) * self.param['T_0'])


        if self.plotting:
            import matplotlib.pyplot as plt

            plt.figure(1)
            plt.plot(meshes[2].coordinates()[:]* self.param['t_0'], np.array(u_fem2_01)*self.param['T_0'], '-or', label='FEM h1')
            plt.plot(meshes[2].coordinates()[:] * self.param['t_0'], np.array(u_fem2_02) * self.param['T_0'], '-or',
                     label='FEM h2')
            plt.plot(meshes[2].coordinates()[:] * self.param['t_0'], upgd_bc_01 * self.param['T_0'],
                     '-*g', label='PGD h1')
            plt.plot(meshes[2].coordinates()[:] * self.param['t_0'], upgd_bc_02 * self.param['T_0'],
                     '-*g', label='PGD h2')
            plt.title(f"solution at [(x,y),h]={self.point},{self.h_fixed} over time")
            plt.xlabel("time t [m]")
            plt.ylabel("Temperature T [°C]")
            plt.legend()

            plt.figure(2) #, figsize=(15, 15), dpi=80)
            plt_field = dolfin.plot(u_fem[-1])
            plt.colorbar(plt_field)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('FEM Temp end time')

            fig, ax = plt.subplots(1,1) #, figsize=(15,15), dpi=80)
            X, Y = np.meshgrid(X,Y)
            cp = ax.contourf(X, Y, Z)
            fig.colorbar(cp)  # Add a colorbar to a plot
            ax.set_title('PGD Temp end time')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            fig, axall = plt.subplots(2,2)
            ax=axall[0,0]
            for i in range(pgd_fd.used_numModes):
                ax.plot(meshes[0].coordinates()[:]*self.param['L_0'],pgd_fd.mesh[0].attributes[0].interpolationfct[i].compute_vertex_values()[:])
            ax.set_title('X modes')

            ax = axall[0, 1]
            for i in range(pgd_fd.used_numModes):
                ax.plot(meshes[1].coordinates()[:] * self.param['L_0'],
                        pgd_fd.mesh[1].attributes[0].interpolationfct[i].compute_vertex_values()[:])
            ax.set_title('Y modes')

            ax = axall[1, 0]
            for i in range(pgd_fd.used_numModes):
                ax.plot(meshes[2].coordinates()[:] * self.param['t_0'],
                        pgd_fd.mesh[2].attributes[0].interpolationfct[i].compute_vertex_values()[:])
            ax.set_title('t modes')

            ax = axall[1, 1]
            for i in range(pgd_fd.used_numModes):
                ax.plot(meshes[3].coordinates()[:],
                        pgd_fd.mesh[3].attributes[0].interpolationfct[i].compute_vertex_values()[:])
            ax.set_title('h modes')

            plt.show()


        self.assertTrue(0.00001 < 1e-3)

if __name__ == '__main__':
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging
    logging.basicConfig(level=logging.INFO)

    unittest.main()