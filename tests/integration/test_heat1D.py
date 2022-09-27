'''
    1D transient thermo problem
    PGD variables: space: x, time: t, source amplitude q

    problem:    strong form: rho cp \partial T/\partial t - k \partial^2 T/\partial x² = q
                weak form: \int T^* rho cp \partial T/\partial t dV + \int \partial T^*/\partial x k \partial T/\partial x dV = \int T^* q dV

                2 cases:
                a:  Q(x,t,q) = (Goldak)
                    T(x,t=0,q) = Tamb (constant)
                b:  Q(x,t,q) = 0
                    T(x,t=0,q) = (Goldak)

    PGD approach: T=sum F(x)F(t)F(q)


'''

import unittest
import dolfin
import numpy as np

from pgdrome.solver import PGDProblem1, FD_matrices

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

def create_bc(Vs,dom,param):
    # boundary conditions list

    # Initial condition
    def init(x, on_boundary):
        return x < 0.0 + 1E-5

    initCond = dolfin.DirichletBC(Vs[1], 0, init)

    return [0, initCond, 0]

def problem_assemble_lhs_FEM(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem

    if typ == 'r':
        a = dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["cp"] * fct_F * var_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * fct_F.dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
    if typ == 's':
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["cp"] * fct_F.dx(0) * var_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * fct_F * var_F * dolfin.dx(meshes[1])
    if typ == 'w':
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1].dx(0) * Fs[1] * dolfin.dx(meshes[1]))) \
            * param["rho"] * param["cp"] * fct_F * var_F * dolfin.dx(meshes[2])\
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))) \
            * param["k"] * fct_F * var_F * dolfin.dx(meshes[2])
    return a

def problem_assemble_rhs_FEM(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem

    IC = [param["IC_x"], param["IC_t"], param["IC_q"]]

    if typ == 'r':
        l = dolfin.Constant(dolfin.assemble(Q[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Q[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * Q[0] * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(dolfin.assemble(IC[1].dx(0) * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(IC[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["cp"] * IC[0] * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(dolfin.assemble(IC[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(IC[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * IC[0].dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
        if nE > 0:
            for old in range(nE):
                l +=- dolfin.Constant(dolfin.assemble(PGD_func[1][old].dx(0) * Fs[1] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["cp"] * PGD_func[0][old] * var_F * dolfin.dx(meshes[0]) \
                    - dolfin.Constant(dolfin.assemble(PGD_func[1][old] * Fs[1] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["k"] * PGD_func[0][old].dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
    if typ == 's':
        l = dolfin.Constant(dolfin.assemble(Q[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Q[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * Q[1] * var_F * dolfin.dx(meshes[1]) \
            - dolfin.Constant(dolfin.assemble(IC[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["cp"] * IC[1].dx(0) * var_F * dolfin.dx(meshes[1]) \
            - dolfin.Constant(dolfin.assemble(IC[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * IC[1] * var_F * dolfin.dx(meshes[1])
        if nE > 0:
            for old in range(nE):
                l +=- dolfin.Constant(dolfin.assemble(PGD_func[0][old] * Fs[0] * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["cp"] * PGD_func[1][old].dx(0) * var_F * dolfin.dx(meshes[1]) \
                    - dolfin.Constant(dolfin.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["k"] * PGD_func[1][old] * var_F * dolfin.dx(meshes[1])
    if typ == 'w':
        l = dolfin.Constant(dolfin.assemble(Q[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Q[1] * Fs[1] * dolfin.dx(meshes[1]))) \
            * Q[2] * var_F * dolfin.dx(meshes[2]) \
            - dolfin.Constant(dolfin.assemble(IC[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC[1].dx(0) * Fs[1] * dolfin.dx(meshes[1]))) \
            * param["rho"] * param["cp"] * IC[2] * var_F * dolfin.dx(meshes[2]) \
            - dolfin.Constant(dolfin.assemble(IC[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC[1] * Fs[1] * dolfin.dx(meshes[1]))) \
            * param["k"] * IC[2] * var_F * dolfin.dx(meshes[2])
        if nE > 0:
            for old in range(nE):
                l +=- dolfin.Constant(dolfin.assemble(PGD_func[0][old] * Fs[0] * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[1][old].dx(0) * Fs[1] * dolfin.dx(meshes[1]))) \
                    * param["rho"] * param["cp"] * PGD_func[2][old] * var_F * dolfin.dx(meshes[2]) \
                    - dolfin.Constant(dolfin.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[1][old] * Fs[1] * dolfin.dx(meshes[1]))) \
                    * param["k"] * PGD_func[2][old] * var_F * dolfin.dx(meshes[2])
    return l

def problem_assemble_lhs_FDtime(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem

    if typ == 'r':
        alpha_1 = Fs[1].vector()[:].transpose() @ param['D1_up_t'] @ Fs[1].vector()[:]
        alpha_2 = Fs[1].vector()[:].transpose() @ param['M_t'] @ Fs[1].vector()[:]
        a = dolfin.Constant(alpha_1 \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["cp"] * fct_F * var_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(alpha_2 \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * fct_F.dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
    if typ == 's':
        a = dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * param["rho"] * param["cp"] * param['D1_up_t'] \
            + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * param["k"] * param['M_t']
        # add initial condition
        a[:, param['bc_idx']] = 0
        a[param['bc_idx'], :] = 0
        a[param['bc_idx'], param['bc_idx']] = 1
    if typ == 'w':
        alpha_1 = Fs[1].vector()[:].transpose() @ param['D1_up_t'] @ Fs[1].vector()[:]
        alpha_2 = Fs[1].vector()[:].transpose() @ param['M_t'] @ Fs[1].vector()[:]
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * alpha_1) \
            * param["rho"] * param["cp"] * fct_F * var_F * dolfin.dx(meshes[2])\
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * alpha_2) \
            * param["k"] * fct_F * var_F * dolfin.dx(meshes[2])
    return a

def problem_assemble_rhs_FDtime(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem

    IC = [param["IC_x"], param["IC_t"], param["IC_q"]]

    if typ == 'r':
        betha_1 = Fs[1].vector()[:].transpose() @ param['M_t'] @ Q[1].vector()[:]
        alpha_1 = Fs[1].vector()[:].transpose() @ param['D1_up_t'] @ IC[1].vector()[:]
        alpha_2 = Fs[1].vector()[:].transpose() @ param['M_t'] @ IC[1].vector()[:]
        l = dolfin.Constant(betha_1 \
            * dolfin.assemble(Q[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * Q[0] * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(alpha_1 \
            * dolfin.assemble(IC[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["cp"] * IC[0] * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(alpha_2 \
            * dolfin.assemble(IC[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * IC[0].dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
        if nE > 0:
            for old in range(nE):
                alpha_old_1 = Fs[1].vector()[:].transpose() @ param['D1_up_t'] @ PGD_func[1][old].vector()[:]
                alpha_old_2 = Fs[1].vector()[:].transpose() @ param['M_t'] @ PGD_func[1][old].vector()[:]
                l +=- dolfin.Constant(alpha_old_1 \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["cp"] * PGD_func[0][old] * var_F * dolfin.dx(meshes[0]) \
                    - dolfin.Constant(alpha_old_2 \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["k"] * PGD_func[0][old].dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
    if typ == 's':
        l = dolfin.assemble(Q[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Q[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * param['M_t'] @ Q[1].vector()[:] \
            - dolfin.assemble(IC[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * param["rho"] * param["cp"] * param['D1_up_t'] @ IC[1].vector()[:] \
            - dolfin.assemble(IC[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * param["k"] * param['M_t'] @ IC[1].vector()[:]
        if nE > 0:
            for old in range(nE):
                l +=- dolfin.assemble(PGD_func[0][old] * Fs[0] * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2])) \
                    * param["rho"] * param["cp"] * param['D1_up_t'] @ PGD_func[1][old].vector()[:] \
                    - dolfin.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2])) \
                    * param["k"] * param['M_t'] @ PGD_func[1][old].vector()[:]
        # add initial condition
        l[param['bc_idx']] = 0
    if typ == 'w':
        betha_1 = Fs[1].vector()[:].transpose() @ param['M_t'] @ Q[1].vector()[:]
        alpha_1 = Fs[1].vector()[:].transpose() @ param['D1_up_t'] @ IC[1].vector()[:]
        alpha_2 = Fs[1].vector()[:].transpose() @ param['M_t'] @ IC[1].vector()[:]
        l = dolfin.Constant(dolfin.assemble(Q[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * betha_1 ) \
            * Q[2] * var_F * dolfin.dx(meshes[2]) \
            - dolfin.Constant(dolfin.assemble(IC[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * alpha_1) \
            * param["rho"] * param["cp"] * IC[2] * var_F * dolfin.dx(meshes[2]) \
            - dolfin.Constant(dolfin.assemble(IC[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * alpha_2) \
            * param["k"] * IC[2] * var_F * dolfin.dx(meshes[2])
        if nE > 0:
            for old in range(nE):
                alpha_old_1 = Fs[1].vector()[:].transpose() @ param['D1_up_t'] @ PGD_func[1][old].vector()[:]
                alpha_old_2 = Fs[1].vector()[:].transpose() @ param['M_t'] @ PGD_func[1][old].vector()[:]
                l +=- dolfin.Constant(dolfin.assemble(PGD_func[0][old] * Fs[0] * dolfin.dx(meshes[0])) \
                    * alpha_old_1) \
                    * param["rho"] * param["cp"] * PGD_func[2][old] * var_F * dolfin.dx(meshes[2]) \
                    - dolfin.Constant(dolfin.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
                    * alpha_old_2) \
                    * param["k"] * PGD_func[2][old] * var_F * dolfin.dx(meshes[2])
    return l

def create_PGD(param={}, vs=[], q=None, _type=None):

    # define nonhomogeneous dirichlet IC
    param.update({'IC_x': dolfin.interpolate(param['IC_x'],vs[0])})
    param.update({'IC_t': dolfin.interpolate(param['IC_t'],vs[1])})
    param.update({'IC_q': dolfin.interpolate(param['IC_q'],vs[2])})

    # define heat source in x, t and q
    q_x = dolfin.interpolate(q, vs[0])
    q_t = dolfin.interpolate(dolfin.Expression('1.0', degree=1),vs[1])
    q_q = dolfin.interpolate(dolfin.Expression('x[0]*Q', Q=param['Q'], degree=1), vs[2])

    if _type == 'FEM':
        ass_rhs = problem_assemble_rhs_FEM
        ass_lhs = problem_assemble_lhs_FEM
        solve_modes = ["FEM", "FEM", "FEM"]

    elif _type == 'FDtime':
        # create FD matrices from meshes
        t_dofs = np.array(vs[1].tabulate_dof_coordinates()[:].flatten())
        t_sort = np.argsort(t_dofs)
        M_t, _, D1_up_t = FD_matrices(t_dofs[t_sort])
        param['M_t'],param['D1_up_t'] = M_t[t_sort,:][:,t_sort], D1_up_t[t_sort,:][:,t_sort]
        param['bc_idx']=np.where(t_dofs==0)[0]
        ass_rhs = problem_assemble_rhs_FDtime
        ass_lhs = problem_assemble_lhs_FDtime
        solve_modes = ["FEM", "FD", "FEM"]
    else:
        ass_rhs = None
        ass_lhs = None
        solve_modes = None
        print('not a valid type')

    pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-XTQ', name_coord=['X', 'T', 'Q'],
                           modes_info=['T', 'Node', 'Scalar'],
                           Vs=vs, dom=0, bc_fct=create_bc, load=[q_x,q_t,q_q],
                           param=param, rhs_fct=ass_rhs,
                           lhs_fct=ass_lhs, probs=['r', 's', 'w'], seq_fp=np.arange(len(vs)),
                           PGD_nmax=20)

    if _type == 'FDtime':
        pgd_prob.MM = [0, param['M_t'], 0]  # for norms!

    pgd_prob.stop_fp = 'norm'
    pgd_prob.max_fp_it = 50
    pgd_prob.tol_fp_it = 1e-5
    # pgd_prob.fp_init = 'randomized'
    pgd_prob.norm_modes = 'stiff'
    pgd_prob.PGD_tol = 1e-5  # 1e-9 as stopping criterion


    pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes)

    print(pgd_prob.simulation_info)
    print('PGD Amplitude', pgd_prob.amplitude)

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance
    
    return pgd_s, param


# reference model FEM in space and backward euler in time
class Reference:
    
    def __init__(self, param={}, vs=[], q=None, x_fixed=None):
        
        self.vs = vs  # Location
        self.param = param  # Parameters
        self.q = q  # source term

        # time points
        self.time_mesh = self.vs[1].mesh().coordinates()[:]
        self.T_n = dolfin.interpolate(self.param["Tamb_fct"], self.vs[0])

        # problem
        self.mesh = self.vs[0].mesh()
        T = dolfin.TrialFunction(self.vs[0])
        v = dolfin.TestFunction(self.vs[0])
        self.dt = dolfin.Constant(1.)
        self.Q = dolfin.Constant(1.)
        self.F = self.param["rho"] * self.param["cp"] * T * v * dolfin.dx() \
            + self.dt * self.param["k"] * dolfin.dot(dolfin.grad(T), dolfin.grad(v)) * dolfin.dx() \
            - (self.dt * self.Q * self.q + self.param["rho"] * self.param["cp"] * self.T_n) * v * dolfin.dx()

        self.fixed_x = x_fixed

    def __call__(self, values):

        # check time mesh for requested time value
        if not np.where(self.time_mesh == values[0])[0]:
            print("ERROR time step not in mesh What to do?")
        self.Q.assign(values[1]*self.param["Q"])
        
        # Time-stepping
        Ttime = []
        Ttmp = dolfin.Function(self.vs[0])
        Ttmp.vector()[:] = 1 * self.T_n.vector()[:]
        Ttime.append(Ttmp)  # otherwise it will be overwritten with new solution
        Txfixed = [np.copy(self.T_n(self.fixed_x))]
        T = dolfin.Function(self.vs[0])
        for i in range(len(self.time_mesh)-1):
            self.dt.assign(self.time_mesh[i+1]-self.time_mesh[i])
            # Compute solution
            a, L = dolfin.lhs(self.F), dolfin.rhs(self.F)
            dolfin.solve(a == L, T)
            # Update previous solution
            self.T_n.assign(T)

            # store solution
            Ttmp = dolfin.Function(self.vs[0])
            Ttmp.vector()[:]=1*T.vector()[:]
            Ttime.append(Ttmp)
            Txfixed.append(np.copy(T(self.fixed_x)))
            
        return Ttime, Txfixed  # solution in time over x and time solution at fixed x

 
# test problem
class problem(unittest.TestCase):
    
    def setUp(self):
        
        # global parameters
        self.param = {"rho": 1, "cp": 1, "k": 0.5, 'Tamb': 25, 'Q': 1,
                      'af': 0.2, 'ar': 0.2, 'xc': 0.5, 'lx': 1, 'lt': 1}  #-comparable matlab code proofed (coarse mesh)

        # self.param = {"rho": 7100, "cp": 3100, "k": 100, 'Q': 5000, 'Tamb': 25,
        #               'af': 0.02, 'ar': 0.02, 'xc': 0.05, 'lx': 0.1, 'lt': 10}  # finer mesh needed
        # self.param = {"rho": 7100, "cp": 3100, "k": 100, 'Q': 100, 'Tamb': 25,
        #               'af': 0.002, 'ar': 0.002, 'xc': 0.05, 'lx': 0.1, 'lt': 10}  # finer mesh needed

        self.ranges = [[0., self.param['lx']],  # xmin, xmax
                       [0., self.param['lt']],  # tmin, tmax
                       [0.5, 1.0]]              # qmin, qmax

        self.ords = [1, 1, 1]  # x, t, q
        self.elems = [15, 10, 10]
        # self.elems = [500, 100, 10]

        # evaluation parameters
        self.fixed_dim = 0
        self.t_fixed = 0.9*self.param['lt']
        self.q_fixed = 1.
        self.x_fixed = 0.5*self.param['lx']

        # self.plotting = True
        self.plotting = False
        
    def TearDown(self):
        pass
    
    def test_heating(self):
        # #case heating
        ff = 6*np.sqrt(3) / ((self.param["af"]+self.param["ar"])*self.param["af"]*self.param["af"]*np.pi**(3/2))
        self.q = dolfin.Expression('ff* exp(-3*(pow(x[0]-xc,2)/pow(af,2)))',
                                   degree=4, ff=ff, af=self.param['af'], ar=self.param['ar'], xc=self.param['xc'])

        self.param['Tamb_fct'] = dolfin.Expression('Tamb', degree=1, Tamb=self.param["Tamb"])  #initial condition FEM
        self.param['IC_t'] = self.param['Tamb_fct']
        self.param['IC_x'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_q'] = dolfin.Expression('1.0', degree=1)
        
        # MESH
        meshes, vs = create_meshes(self.elems, self.ords, self.ranges)
        
        # PGD
        pgd_fem, param = create_PGD(param=self.param, vs=vs, q=self.q, _type="FEM")
        pgd_fd, param = create_PGD(param=self.param, vs=vs, q=self.q, _type="FDtime")

        # error at given values:
        # FEM reference solution 2D Problem at given values self.values
        tidx = np.where(meshes[1].coordinates()[:] == self.t_fixed)[0][0]
        u_fem, u_fem2 = Reference(param=self.param, vs=vs, q=self.q, x_fixed=self.x_fixed)(
            [self.ranges[1][1], self.q_fixed])

        upgd_fem = pgd_fem.evaluate(self.fixed_dim, [1, 2], [self.t_fixed,self.q_fixed], 0)
        upgd_fem_bc = upgd_fem.compute_vertex_values()[:] + \
                     param['IC_x'].compute_vertex_values()[:] * param['IC_t'](self.t_fixed) * param["IC_q"](
           self.q_fixed)

        upgd_fd = pgd_fd.evaluate(self.fixed_dim, [1, 2], [self.t_fixed, self.q_fixed], 0)
        upgd_fd_bc = upgd_fd.compute_vertex_values()[:] + \
                     param['IC_x'].compute_vertex_values()[:] * param['IC_t'](self.t_fixed) * param["IC_q"](
            self.q_fixed)

        errors_FEM11 = np.linalg.norm(upgd_fd_bc - u_fem[tidx].compute_vertex_values()[:]) / np.linalg.norm(u_fem[tidx].compute_vertex_values()[:])  # PGD FD - FEM
        errors_FEM12 = np.linalg.norm(upgd_fem_bc - u_fem[tidx].compute_vertex_values()[:]) / np.linalg.norm(u_fem[tidx].compute_vertex_values()[:])  # PGD FEM - FEM
        print('error in space', errors_FEM11)

        # solution at fixed place over time
        upgd_fem2 = pgd_fem.evaluate(1, [0, 2], [self.x_fixed,self.q_fixed], 0)
        upgd_fem2_bc = upgd_fem2.compute_vertex_values()[:] + \
                     param['IC_x'](self.x_fixed) * param['IC_t'].compute_vertex_values()[:] * param["IC_q"](
           self.q_fixed)
        upgd_fd2 = pgd_fd.evaluate(1, [0, 2], [self.x_fixed, self.q_fixed], 0)
        upgd_fd2_bc = upgd_fd2.compute_vertex_values()[:] + \
                      param['IC_x'](self.x_fixed) * param['IC_t'].compute_vertex_values()[:] * param["IC_q"](
            self.q_fixed)

        errors_FEM21 = np.linalg.norm(upgd_fd2_bc - u_fem2) / np.linalg.norm(u_fem2)  # PGD FD - FEM
        errors_FEM22 = np.linalg.norm(upgd_fem2_bc - u_fem2) / np.linalg.norm(u_fem2)  # PGD FEM - FEM
        print('error in time', errors_FEM21)

        if self.plotting:
            #### plotting optional
            import matplotlib.pyplot as plt

            # Temperature over space at specific time
            plt.figure(1)
            plt.plot(meshes[0].coordinates()[:], u_fem[tidx].compute_vertex_values()[:], '-or', label=f'FEM')
            plt.plot(upgd_fem.function_space().mesh().coordinates()[:], upgd_fem_bc, '-*b',
                    label=f"PGD FEM t end")
            plt.plot(upgd_fd.function_space().mesh().coordinates()[:], upgd_fd_bc, '-*g',
                     label=f"PGD FDtime t end")
            plt.title(f"PGD solution over space")
            plt.xlabel("Space x [m]")
            plt.ylabel("Temperature T [°C]")
            plt.legend()

            plt.figure(2)
            plt.plot(meshes[1].coordinates()[:], u_fem2, '-or', label='FEM')
            plt.plot(meshes[1].coordinates()[:], upgd_fem2_bc, '-*b', label='PGD FEM')
            plt.plot(meshes[1].coordinates()[:], upgd_fd2_bc, '-*g', label='PGD FD')
            plt.title(f"PGD solution at [x,q]={self.x_fixed},{self.q_fixed} over time")
            plt.xlabel("time t [m]")
            plt.ylabel("Temperature T [°C]")
            plt.legend()

            plt.draw()
            plt.show()

        # for mesh discr: [15, 10, 10] //  and first parameter set. Finer mesh leads to lower errors
        self.assertTrue(errors_FEM11 < 1e-3)  # 1e-6
        self.assertTrue(errors_FEM21 < 1e-2)  # 5e-4
        self.assertTrue(errors_FEM12 < 1e-3)  # 1e-6
        self.assertTrue(errors_FEM22 < 1e-2)  # 5e-4

    def test_cooling(self):
        # case cooling!
        self.q = dolfin.Expression('0', degree=1)
        vf_a = 6*np.sqrt(3) / (2*self.param['af']**3*np.pi**(3/2))
        self.param['Tamb_fct'] = dolfin.Expression(
            'vf* exp(-3*(pow(x[0]-xc,2)/pow(a,2)))',
            degree=4, vf=self.q_fixed*vf_a, a=self.param['af'], xc=self.param['xc'])  # initial condition fem
        self.param['IC_t'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_x'] = dolfin.Expression(
                'vf* exp(-3*(pow(x[0]-xc,2)/pow(a,2)))',
                degree=4, vf=vf_a, a=self.param['af'], xc=self.param['xc'])
        self.param['IC_q'] = dolfin.Expression('x[0]', degree=1)

         # MESH
        meshes, vs = create_meshes(self.elems, self.ords, self.ranges)
        # PGD
        pgd_fd, param = create_PGD(param=self.param, vs=vs, q=self.q, _type="FDtime")

        # error computation at fixed values
        # FEM reference solution 2D Problem at given values self.values
        tidx = np.where(meshes[1].coordinates()[:] == self.t_fixed)[0][0]
        u_fem, u_fem2 = Reference(param=self.param, vs=vs, q=self.q, x_fixed=self.x_fixed)(
            [self.ranges[1][1], self.q_fixed])

        upgd_fd = pgd_fd.evaluate(self.fixed_dim, [1, 2], [self.t_fixed, self.q_fixed], 0)
        upgd_fd_bc = upgd_fd.compute_vertex_values()[:] + \
                     param['IC_x'].compute_vertex_values()[:] * param['IC_t'](self.t_fixed) * param["IC_q"](
            self.q_fixed)
        errors_FEM11 = np.linalg.norm(upgd_fd_bc - u_fem[tidx].compute_vertex_values()[:]) / np.linalg.norm(
            u_fem[tidx].compute_vertex_values()[:])  # PGD FD - FEM
        print('error in space', errors_FEM11)

        # solution at fixed place over time
        upgd_fd2 = pgd_fd.evaluate(1, [0, 2], [self.x_fixed, self.q_fixed], 0)
        upgd_fd2_bc = upgd_fd2.compute_vertex_values()[:] + \
                      param['IC_x'](self.x_fixed) * param['IC_t'].compute_vertex_values()[:] * param["IC_q"](
            self.q_fixed)

        errors_FEM21 = np.linalg.norm(upgd_fd2_bc - u_fem2) / np.linalg.norm(u_fem2)  # PGD FD - FEM
        print('error in time', errors_FEM21)

        if self.plotting:
            #### plotting optional
            import matplotlib.pyplot as plt


            # Temperature over space at specific time
            plt.figure(1)
            plt.plot(meshes[0].coordinates()[:], u_fem[tidx].compute_vertex_values()[:], '-or', label=f'FEM')
            plt.plot(upgd_fd.function_space().mesh().coordinates()[:], upgd_fd_bc, '-*g',
                     label=f"PGD FDtime t end")
            plt.title(f"PGD solution over space")
            plt.xlabel("Space x [m]")
            plt.ylabel("Temperature T [°C]")
            plt.legend()

            plt.figure(2)
            plt.plot(meshes[1].coordinates()[:], u_fem2, '-or', label='FEM')
            plt.plot(meshes[1].coordinates()[:], upgd_fd2_bc, '-*g', label='PGD FD')
            plt.title(f"PGD solution at [x,q]={self.x_fixed},{self.q_fixed} over time")
            plt.xlabel("time t [m]")
            plt.ylabel("Temperature T [°C]")
            plt.legend()

            plt.draw()
            plt.show()

        # for mesh discr: [15, 10, 10] and first parameter set!
        self.assertTrue(errors_FEM11 < 1e-6)
        self.assertTrue(errors_FEM21 < 5e-6)
        
if __name__ == '__main__':
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging
    logging.basicConfig(level=logging.INFO)

    unittest.main()
