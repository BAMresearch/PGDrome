'''
    simple 1D PGD example (heat equation with a point heat source) with three PGD variables (space, time and heat input)
    solving PGD problem in standard way using FEM
    returning PGDModel (as forward model) or PGD instance
'''

import unittest
import dolfin
import fenics
import numpy as np
import matplotlib.pyplot as plt

from pgdrome.solver import PGDProblem1, FD_matrices
from pgdrome.model import PGDErrorComputation

#TODO: delete!

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

def problem_assemble_lhs_FEM(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem

    if typ == 'x':
        a = dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * fct_F * var_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * fct_F.dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
    if typ == 't':
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * fct_F.dx(0) * var_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * fct_F * var_F * dolfin.dx(meshes[1])
    if typ == 'w':
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1].dx(0) * Fs[1] * dolfin.dx(meshes[1]))) \
            * param["rho"] * param["c_p"] * fct_F * var_F * dolfin.dx(meshes[2])\
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))) \
            * param["k"] * fct_F * var_F * dolfin.dx(meshes[2])
    return a

def problem_assemble_rhs_FEM(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem

    IC_x = param["IC_x"]
    IC_t = param["IC_t"]
    IC_Eta = param["IC_Eta"]       
                
    if typ == 'x':
        l = dolfin.Constant(dolfin.assemble(Q[1][0] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Q[2][0] * Fs[2] * dolfin.dx(meshes[2]))) \
            * Q[0][0] * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(dolfin.assemble(Fs[1] * IC_t.dx(0) * dolfin.dx(meshes[1])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * IC_x * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(dolfin.assemble(Fs[1] * IC_t * dolfin.dx(meshes[1])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * IC_x.dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
        if nE > 0:
            for old in range(nE):
                l +=- dolfin.Constant(dolfin.assemble(PGD_func[1][old].dx(0) * Fs[1] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["c_p"] * PGD_func[0][old] * var_F * dolfin.dx(meshes[0]) \
                    - dolfin.Constant(dolfin.assemble(PGD_func[1][old] * Fs[1] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["k"] * PGD_func[0][old].dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
                    
    if typ == 't':
        l = dolfin.Constant(dolfin.assemble(Q[0][0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Q[2][0] * Fs[2] * dolfin.dx(meshes[2]))) \
            * Q[1][0] * var_F * dolfin.dx(meshes[1]) \
            - dolfin.Constant(dolfin.assemble(IC_x * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * var_F * IC_t.dx(0) * dolfin.dx(meshes[1]) \
            - dolfin.Constant(dolfin.assemble(IC_x.dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * var_F * IC_t * dolfin.dx(meshes[1])
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
            - dolfin.Constant(dolfin.assemble(IC_x * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * IC_t.dx(0) * dolfin.dx(meshes[1]))) \
            * param["rho"] * param["c_p"] * IC_Eta * var_F * dolfin.dx(meshes[2]) \
            - dolfin.Constant(dolfin.assemble(IC_x.dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * IC_t * dolfin.dx(meshes[1]))) \
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

def problem_assemble_lhs_FD(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem

    alpha_t1 = Fs[1].vector()[:].transpose() @ param['D1_up'] @ Fs[1].vector()[:]
    alpha_t2 = Fs[1].vector()[:].transpose() @ param['M1'] @ Fs[1].vector()[:]

    if typ == 'x':        
        a = dolfin.Constant(alpha_t1 \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * fct_F * var_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(alpha_t2 \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * fct_F.dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
            
    if typ == 't':
        a = dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * param["rho"] * param["c_p"] * param['D1_up'] \
            + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * param["k"] * param['M1']
            
        # add initial condition
        a[:,param['bc_idx']]=0
        a[param['bc_idx'],:] = 0
        a[param['bc_idx'], param['bc_idx']] = 1
        
    if typ == 'w':
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * alpha_t1) \
            * param["rho"] * param["c_p"] * fct_F * var_F * dolfin.dx(meshes[2])\
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * alpha_t2) \
            * param["k"] * fct_F * var_F * dolfin.dx(meshes[2])
            
    return a

def problem_assemble_rhs_FD(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem

    IC_x = param["IC_x"]
    IC_t = param["IC_t"]
    IC_Eta = param["IC_Eta"]
    
    beta_t1 = Fs[1].vector()[:].transpose() @ param['M1'] @ Q[1][0].vector()[:]
    beta_t2 = Fs[1].vector()[:].transpose() @ param['D1_up'] @ IC_t.vector()[:]
    beta_t3 = Fs[1].vector()[:].transpose() @ param['M1'] @ IC_t.vector()[:]

    if typ == 'x':
        l = dolfin.Constant(beta_t1 \
            * dolfin.assemble(Q[2][0] * Fs[2] * dolfin.dx(meshes[2]))) \
            * Q[0][0] * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(beta_t2 \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["rho"] * param["c_p"] * IC_x * var_F * dolfin.dx(meshes[0]) \
            - dolfin.Constant(beta_t3 \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2]))) \
            * param["k"] * IC_x.dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
        if nE > 0:
            for old in range(nE):
                alpha_t1 = Fs[1].vector()[:].transpose() @ param['D1_up'] @ PGD_func[1][old].vector()[:]
                alpha_t2 = Fs[1].vector()[:].transpose() @ param['M1'] @ PGD_func[1][old].vector()[:]
                
                l +=- dolfin.Constant(alpha_t1 \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["c_p"] * PGD_func[0][old] * var_F * dolfin.dx(meshes[0]) \
                    - dolfin.Constant(alpha_t2 \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2]))) \
                    * param["k"] * PGD_func[0][old].dx(0) * var_F.dx(0) * dolfin.dx(meshes[0])
                    
    if typ == 't':
        l = dolfin.assemble(Q[0][0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Q[2][0] * Fs[2] * dolfin.dx(meshes[2])) \
            * param['M1'] @ Q[1][0].vector()[:] \
            - dolfin.assemble(IC_x * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2])) \
            * param["rho"] * param["c_p"] * param['D1_up'] @ IC_t.vector()[:] \
            - dolfin.assemble(IC_x.dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(IC_Eta * Fs[2] * dolfin.dx(meshes[2])) \
            * param["k"] * param['M1'] @ IC_t.vector()[:]
        if nE > 0:
            for old in range(nE):
                l +=- dolfin.assemble(PGD_func[0][old] * Fs[0] * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2])) \
                    * param["rho"] * param["c_p"] * param['D1_up'] @ PGD_func[1][old].vector()[:] \
                    - dolfin.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
                    * dolfin.assemble(PGD_func[2][old] * Fs[2] * dolfin.dx(meshes[2])) \
                    * param["k"] * param['M1'] @ PGD_func[1][old].vector()[:]
                
        # add initial condition
        l[param['bc_idx']]=0
                    
    if typ == 'w':
        l = dolfin.Constant(dolfin.assemble(Q[0][0] * Fs[0] * dolfin.dx(meshes[0])) \
            * beta_t1) \
            * Q[2][0] * var_F * dolfin.dx(meshes[2]) \
            - dolfin.Constant(dolfin.assemble(IC_x * Fs[0] * dolfin.dx(meshes[0])) \
            * beta_t2) \
            * param["rho"] * param["c_p"] * IC_Eta * var_F * dolfin.dx(meshes[2]) \
            - dolfin.Constant(dolfin.assemble(IC_x.dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * beta_t3) \
            * param["k"] * IC_Eta * var_F * dolfin.dx(meshes[2])
        if nE > 0:
            for old in range(nE):
                alpha_t1 = Fs[1].vector()[:].transpose() @ param['D1_up'] @ PGD_func[1][old].vector()[:]
                alpha_t2 = Fs[1].vector()[:].transpose() @ param['M1'] @ PGD_func[1][old].vector()[:]
                
                l +=- dolfin.Constant(dolfin.assemble(PGD_func[0][old] * Fs[0] * dolfin.dx(meshes[0])) \
                    * alpha_t1) \
                    * param["rho"] * param["c_p"] * PGD_func[2][old] * var_F * dolfin.dx(meshes[2]) \
                    - dolfin.Constant(dolfin.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
                    * alpha_t2) \
                    * param["k"] * PGD_func[2][old] * var_F * dolfin.dx(meshes[2])
                    
    return l

def main_FEM(vs, params, name=None):
    '''computation of PGD solution for given problem '''

    # define some parameters
    param = params
    
    # define nonhomogeneous dirichlet IC in time t
    param.update({'IC_x': dolfin.interpolate(dolfin.Expression('1.0', degree=4), vs[0])})
    param.update({'IC_t': dolfin.interpolate(
        dolfin.Expression('(x[0] < 0.0 + 1E-8) ? Tamb : 0', degree=4, Tamb=params['T_amb']), vs[1])})
    param.update({'IC_Eta': dolfin.interpolate(dolfin.Expression('1.0', degree=4), vs[2])})

    # define heat source in x, t and eta
    q1 = [dolfin.Expression('x[0] < 0.05 - af + DOLFIN_EPS ? p1 : (x[0] > 0.05 + af - DOLFIN_EPS ? p1 : p2)', degree=4, af=param['af'], p1=0, p2=param['P'])]
    q2 = [dolfin.Expression('1.0', degree=1)]
    q3 = [dolfin.Expression('x[0]', degree=1)]

    prob = ['x', 't', 'w'] # problems according problem_assemble_fcts
    seq_fp = np.arange(len(vs))  # default sequence of Fixed Point iteration
    PGD_nmax = 15      # max number of PGD modes

    pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-XTEta', name_coord=['X', 'T', 'Eta'],
                           modes_info=['T_x', 'Node', 'Scalar'],
                           Vs=vs, dom=0, bc_fct=create_bc, load=[q1,q2,q3],
                           param=param, rhs_fct=problem_assemble_rhs_FEM,
                           lhs_fct=problem_assemble_lhs_FEM, probs=prob, seq_fp=seq_fp,
                           PGD_nmax=PGD_nmax)

    # possible solver parameters (if not given then default values will be used!)
    pgd_prob.stop_fp = 'chady'
    # pgd_prob.stop_fp = 'norm'
    pgd_prob.max_fp_it = 50
    pgd_prob.tol_fp_it = 1e-5
    # pgd_prob.fp_init = 'randomized'

    pgd_prob.solve_PGD(_problem='linear')
    print(pgd_prob.simulation_info)
    print('PGD Amplitude', pgd_prob.amplitude)

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance
    
    return pgd_s, param

def main_FD(vs, params):
    '''computation of PGD solution for given problem '''

    # define some parameters
    param = params

    # define nonhomogeneous dirichlet IC in time t
    param.update({'IC_x': dolfin.interpolate(dolfin.Expression('1.0', degree=4), vs[0])})
    param.update({'IC_t': dolfin.interpolate(
        dolfin.Expression('(x[0] < 0.0 + 1E-8) ? Tamb : 0', degree=4, Tamb=params['T_amb']), vs[1])})
    param.update({'IC_Eta': dolfin.interpolate(dolfin.Expression('1.0', degree=4), vs[2])})
    
    # define heat source in x, t and eta
    q1 = [dolfin.Expression('x[0] < 0.05 - af + DOLFIN_EPS ? p1 : (x[0] > 0.05 + af - DOLFIN_EPS ? p1 : p2)', degree=4, af=param['af'], p1=0, p2=param['P'])]
    q2 = [dolfin.interpolate(dolfin.Expression('1.0', degree=1), vs[1])]
    q3 = [dolfin.interpolate(dolfin.Expression('x[0]', degree=1), vs[2])]

    prob = ['x', 't', 'w'] # problems according problem_assemble_fcts
    seq_fp = np.arange(len(vs))  # default sequence of Fixed Point iteration
    PGD_nmax = 15      # max number of PGD modes

    pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-XTEta', name_coord=['X', 'T', 'Eta'],
                           modes_info=['T_x', 'Node', 'Scalar'],
                           Vs=vs, dom=0, bc_fct=create_bc, load=[q1,q2,q3],
                           param=param, rhs_fct=problem_assemble_rhs_FD,
                           lhs_fct=problem_assemble_lhs_FD, probs=prob, seq_fp=seq_fp,
                           PGD_nmax=PGD_nmax)

    # possible solver parameters (if not given then default values will be used!)
    pgd_prob.stop_fp = 'chady'
    # pgd_prob.stop_fp = 'norm'
    pgd_prob.max_fp_it = 50
    pgd_prob.tol_fp_it = 1e-5 
    # pgd_prob.fp_init = 'randomized'

    pgd_prob.solve_PGD(_problem='linear', solve_modes=["FEM","FD","FEM"])
    print(pgd_prob.simulation_info)
    print('PGD Amplitude', pgd_prob.amplitude)

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance
    
    return pgd_s, param


# TEST: PGD result VS FEM result
#==============================================================================  

# Finite Element Model
#=======================================
class Reference_solution():
    
    def __init__(self,Vs=[], param=[], meshes=[], x_fixed=[]):
        
        self.Vs = Vs # Location
        self.param = param # Parameters
        self.meshes = meshes # Meshes
        self.x_fixed = x_fixed
        
        # Dirichlet BC:
        self.bc = create_bc(self.Vs,0,self.param)
        
    def fem_definition(self,t_max, eta):        
        rho = self.param['rho']                                 # material density [kg/m³]
        k = self.param['k']                                     # heat conductivity [W/m°C]
        cp = self.param['c_p']                                  # specific heat capacity [J/kg°C]
        T_amb = self.param['T_amb']
        time_points = np.linspace(0, t_max, num=len(self.meshes[1].coordinates()))
        dt = time_points[1]-time_points[0]
        # Define initial value
        T_n = fenics.project(fenics.Expression("T_amb", domain=self.meshes[0], degree=4, T_amb=T_amb), self.Vs[0])
        # Define heat input         
        q = fenics.Expression('x[0] < 0.05 - af + DOLFIN_EPS ? p1 : (x[0] > 0.05 + af - DOLFIN_EPS ? p1 : p2)', degree=4, af=self.param['af'], p1=0, p2=eta*self.param['P'])
        # Define problem functions
        T = fenics.TrialFunction(self.Vs[0])
        v = fenics.TestFunction(self.Vs[0])

        # Collect variational form
        F =  rho*cp*T*v*fenics.dx \
            + dt*k*fenics.dot(fenics.grad(T), fenics.grad(v))*fenics.dx \
            - (dt*q + rho*cp*T_n)*v*fenics.dx
        a, L = fenics.lhs(F), fenics.rhs(F)
        
        # Time-stepping
        T = fenics.Function(self.Vs[0])
        for n in range(len(time_points)-1):
            # Compute solution
            fenics.solve(a == L, T)
            # Update previous solution
            T_n.assign(T)
            
        # If specific points are given
        if self.x_fixed:
            T_out = np.zeros((len(self.x_fixed),self.meshes[0].topology().dim()))
            for i in range(len(self.x_fixed)):
                T_out[i,:]=np.array(T(self.x_fixed[i]))
            return T_out
        else:
            return T # return full vector
        
        # return T
            
    def __call__(self, data_test):
        
        # sampled variable values
        t_max = data_test[0]    # last time point
        eta = data_test[1]      # arc efficiency
                
        ref_sol = self.fem_definition(t_max, eta)
        
        return ref_sol
 
# PGD model and Error computation
#=======================================
class PGDproblem(unittest.TestCase):
    
    def setUp(self):
        
        # global parameters
        self.ord = 1  # 1 # 2 # order for each mesh
        self.ords = [self.ord, self.ord, self.ord]
        self.ranges = [[0., 0.1],  # xmin, xmax
                  [0., 10.],  # tmin, tmax
                  [0.7, 0.9]]  # etamin, etamax
        self.num_elem = [1000, # number of elements in x
                         500, # number of elements in t
                         50] # number of elements in eta

        # sampling parameters
        self.fixed_dim = [0] # Fixed variable
        self.n_samples = 10 # Number of samples

        self.param = {"rho": 7100, "c_p": 3100, "k": 100, "P": 10e9, "T_amb": 25, "af": 0.02}

    def TearDown(self):
        pass
    
    def test_solver(self):
        
        # MESH
        #======================================================================
        meshes, Vs = create_meshes([1000, 500, 50], self.ords, self.ranges)
        
        # create FD matrices for time mesh
        # sort dof coordinates and create FD matrices
        x_dofs = np.array(Vs[1].tabulate_dof_coordinates()[:].flatten())
        idx_sort = np.argsort(x_dofs)
        Mt, _, D1_upt = FD_matrices(Vs[1].tabulate_dof_coordinates()[idx_sort])
        # idx for initial condition of time problem
        bc_idx = np.where(x_dofs == 0)

        # store re_sorted according dofs!
        self.param['M1'] = Mt[idx_sort,:][:,idx_sort]
        self.param['D1_up'] = D1_upt[idx_sort, :][:, idx_sort]
        self.param['bc_idx'] = bc_idx[0]
        
        # Computing solution and error
        #======================================================================
        # Solve Reduced-order model: PGD
        pgd_test_FEM, param = main_FEM(Vs, self.param) # Computing PGD with FEM
        pgd_test_FD, param = main_FD(Vs, self.param) # Computing PGD with FD

        # Solve Full-order model: FEM
        fun_FOM = Reference_solution(Vs=Vs, param=self.param, meshes=meshes) # Computing Full-Order model: FEM
        
        # Compute error with FEM
        error_uPGD_FEM = PGDErrorComputation(fixed_dim = self.fixed_dim,
                                          n_samples = self.n_samples,
                                          FOM_model = fun_FOM,
                                          PGD_model = pgd_test_FEM
                                          )
        
        errorL2_FEM, mean_errorL2_FEM, max_errorL2_FEM = error_uPGD_FEM.evaluate_error() # Computing Error
        
        # Compute error with FD
        error_uPGD_FD = PGDErrorComputation(fixed_dim = self.fixed_dim,
                                          n_samples = self.n_samples,
                                          FOM_model = fun_FOM,
                                          PGD_model = pgd_test_FD
                                          )
        
        errorL2_FD, mean_errorL2_FD, max_errorL2_FD = error_uPGD_FD.evaluate_error() # Computing Error
        
        print('Mean error PGD with FEM ', mean_errorL2_FEM)
        print('Max. error PGD with FEM ', max_errorL2_FEM)
        
        print('Mean error PGD with FD ', mean_errorL2_FD)
        print('Max. error PGD with FD ', max_errorL2_FD)
        
        # self.assertTrue(mean_errorL2_FEM<0.01)
        # self.assertTrue(mean_errorL2_FD<0.01)
        
        
        # Computing solution and error
        #======================================================================
        # Create variables array:
        x_test = [[0.05]]  # x (fixed variable)
        data_test = [[1., 0.8],[5.,0.8],[10.,0.8]] # t, eta
        
        # Solve Full-order model: FEM
        fun_FOM2 = Reference_solution(Vs=Vs, param=param, meshes=meshes, x_fixed=x_test) # Computing Full-Order model: FEM

        # Compute error with FEM
        error_uPGD2_FEM = PGDErrorComputation(fixed_dim = self.fixed_dim,
                                          FOM_model = fun_FOM2,
                                          PGD_model = pgd_test_FEM,
                                          data_test = data_test,
                                          fixed_var = x_test
                                          )

        error2_FEM, mean_error2_FEM, max_error2_FEM = error_uPGD2_FEM.evaluate_error()  
        
        # Compute error with FD
        error_uPGD2_FD = PGDErrorComputation(fixed_dim = self.fixed_dim,
                                          FOM_model = fun_FOM2,
                                          PGD_model = pgd_test_FD,
                                          data_test = data_test,
                                          fixed_var = x_test
                                          )

        error2_FD, mean_error2_FD, max_error2_FD = error_uPGD2_FD.evaluate_error()  
        
        # Plot solution over space at specific time
        plt.figure()
        u_fem1 = fun_FOM(data_test[0])
        u_pgd1_FEM = pgd_test_FEM.evaluate(0, [1, 2], [data_test[0][0], data_test[0][1]], 0)
        u_pgd1_FD = pgd_test_FD.evaluate(0, [1, 2], [data_test[0][0], data_test[0][1]], 0)
        u_fem2 = fun_FOM(data_test[-1])
        u_pgd2_FEM = pgd_test_FEM.evaluate(0, [1, 2], [data_test[-1][0], data_test[-1][1]], 0)
        u_pgd2_FD = pgd_test_FD.evaluate(0, [1, 2], [data_test[-1][0], data_test[-1][1]], 0)
        plt.plot(pgd_test_FEM.mesh[0].dataX, u_pgd1_FEM.compute_vertex_values()[:], '-*b', label=f"PGD with FEM at {data_test[0]}s")
        plt.plot(pgd_test_FD.mesh[0].dataX, u_pgd1_FD.compute_vertex_values()[:], '-d', label=f"PGD with FD at {data_test[0]}s")
        plt.plot(pgd_test_FEM.mesh[0].dataX, u_fem1.compute_vertex_values()[:], '-or', label='FEM')
        plt.plot(pgd_test_FEM.mesh[0].dataX, u_pgd2_FEM.compute_vertex_values()[:], '-*g', label=f"PGD with FEM at {data_test[-1]}s")
        plt.plot(pgd_test_FD.mesh[0].dataX, u_pgd2_FD.compute_vertex_values()[:], '-d', label=f"PGD with FD at {data_test[-1]}s")
        plt.plot(pgd_test_FEM.mesh[0].dataX, u_fem2.compute_vertex_values()[:], '-oy', label='FEM')
        plt.title(f"PGD solution at {data_test[0][0]}s over space")
        plt.xlabel("Space x [m]")
        plt.ylabel("Temperature T [°C]")
        plt.legend()
        plt.draw()
        plt.show()
        
        print('Mean error PGD with FEM ', mean_errorL2_FEM)
        print('Max. error PGD with FEM ', max_errorL2_FEM)
        
        print('Mean error PGD with FD ', mean_errorL2_FD)
        print('Max. error PGD with FD ', max_errorL2_FD)
        
        self.assertTrue(mean_error2_FEM<0.01)
        self.assertTrue(mean_error2_FD<0.01)
        
if __name__ == '__main__':
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging
    logging.basicConfig(level=logging.INFO)

    unittest.main()
