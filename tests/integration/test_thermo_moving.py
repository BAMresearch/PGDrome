'''
    1D PGD example (heat equation with a moving goldak heat source) with three PGD variables (space, time and heat input)
    solving PGD problem in standard way using FD in time and FEM else
    returning PGDModel (as forward model) or PGD instance

'''

import unittest
import dolfin
import fenics
import numpy as np
import matplotlib.pyplot as plt

from pgdrome.solver import PGDProblem1, FD_matrices


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

def create_dom(Vs,param):

    #create domains in s
    subdomains = dolfin.MeshFunction("size_t", Vs[0].mesh(), Vs[0].mesh().topology().dim())  # same dimension
    subdomains.set_all(0)

    class left_dom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x < 1 + 1e-8

    class middle_dom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x >= 1 - 1e-8 and x <= 2 + 1e-8
        
    class right_dom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x > 2 - 1e-8

    Left_dom = left_dom()
    Left_dom.mark(subdomains, 0)
    Middle_dom = middle_dom()
    Middle_dom.mark(subdomains, 1)
    Right_dom = right_dom()
    Right_dom.mark(subdomains, 2)

    dom = [subdomains]
    dom.extend(np.zeros(len(Vs)-1,dtype=int))

    return dom

def create_bc(Vs,dom,param):      
    # Initial condition
    def init(x, on_boundary):
        return x < param["r_0"] + 1E-8

    initCond = dolfin.DirichletBC(Vs[1], dolfin.Constant(0.), init)

    return [0, initCond, 0]

def problem_assemble_lhs_FEM(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem
    
    dx_s = dolfin.Measure('dx', domain=meshes[0], subdomain_data=dom[0])

    # define expressions
    Bt = param["Bt"] 
    Bx = param["Bx"] 
    det_J = param["det_J"] 
    
    a = 0
    if typ == 's':
        for i in range(3):
            a +=  dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * det_J[i] * Bt[i][0][1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
                * param["rho"] * param["c_p"] * var_F * Bt[i][0][0] * fct_F.dx(0) * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1].dx(0) * det_J[i] * Bt[i][1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
                * param["rho"] * param["c_p"] * var_F * fct_F * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * det_J[i] * Bx[i] * Bx[i] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
                * param["k"] * var_F.dx(0) * fct_F.dx(0) * dx_s(i)
                
    if typ == 'r':
        for i in range(3):
            a +=  dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][0][0] * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
                * var_F * fct_F * det_J[i] * Bt[i][0][1] * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
                * var_F * fct_F.dx(0) * det_J[i] * Bt[i][1] * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(param["k"] * Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
                * var_F * fct_F * det_J[i] * Bx[i] * Bx[i] * dolfin.dx(meshes[1]) 
                
    if typ == 'Q':
        for i in range(3):
            a +=  dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][0][0] * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * det_J[i] * Bt[i][0][1] * dolfin.dx(meshes[1]))) \
                * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1].dx(0) * det_J[i] * Bt[i][1] * dolfin.dx(meshes[1]))) \
                * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(param["k"] * Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * det_J[i] * Bx[i] * Bx[i] * dolfin.dx(meshes[1]))) \
                * var_F * fct_F * dolfin.dx(meshes[2])
                
    return a

def problem_assemble_rhs_FEM(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem
    
    dx_s = dolfin.Measure('dx', domain=meshes[0], subdomain_data=dom[0])

    # define expressions
    Bt = param["Bt"] 
    Bx = param["Bx"] 
    det_J = param["det_J"]
    IC_s = param["IC_s"] 
    IC_r = param["IC_r"] 
    IC_Q = param["IC_Q"]

    if typ == 's':
            l =   dolfin.Constant(dolfin.assemble(Fs[1] * Q[1][0] * det_J[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Q[2][0] * dolfin.dx(meshes[2]))) \
                * var_F * Q[0][0] * dx_s(1)
            for i in range(0,3):
                l +=- dolfin.Constant(dolfin.assemble(Fs[1] * IC_r * det_J[i] * Bt[i][0][1] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["c_p"] * var_F * Bt[i][0][0] * IC_s.dx(0) * dx_s(i) \
                    - dolfin.Constant(dolfin.assemble(Fs[1] * IC_r.dx(0) * det_J[i] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["c_p"] * var_F * Bt[i][1] * IC_s * dx_s(i) \
                    - dolfin.Constant(dolfin.assemble(Fs[1] * Bx[i] * Bx[i] * IC_r * det_J[i] * dolfin.dx(meshes[1])) \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2]))) \
                    * param["k"] * var_F.dx(0) * IC_s.dx(0) * dx_s(i) 
            if nE > 0:
                for old in range(nE):
                    for j in range(0,3):
                        l +=- dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * det_J[j] * Bt[j][0][1] * dolfin.dx(meshes[1])) \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                            * param["rho"] * param["c_p"] * var_F * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j) \
                            - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old].dx(0) * det_J[j] * dolfin.dx(meshes[1])) \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                            * param["rho"] * param["c_p"] * var_F * Bt[j][1] * PGD_func[0][old] * dx_s(j) \
                            - dolfin.Constant(dolfin.assemble(Fs[1] * Bx[j] * Bx[j] * PGD_func[1][old] * det_J[j] * dolfin.dx(meshes[1])) \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                            * param["k"] * var_F.dx(0) * PGD_func[0][old].dx(0) * dx_s(j) 
                            
    if typ == 'r':
            l =   dolfin.Constant(dolfin.assemble(Fs[0] * Q[0][0] * dx_s(1)) \
                * dolfin.assemble(Fs[2] * Q[2][0] * dolfin.dx(meshes[2]))) \
                * var_F * Q[1][0] * det_J[1] * dolfin.dx(meshes[1])
            for i in range(0,3):
                l +=- dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][0][0] * IC_s.dx(0) * dx_s(i)) \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2]))) \
                    * var_F * IC_r * det_J[i] * Bt[i][0][1] * dolfin.dx(meshes[1]) \
                    - dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][1] * IC_s * dx_s(i)) \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2]))) \
                    * var_F * IC_r.dx(0) * det_J[i] * dolfin.dx(meshes[1]) \
                    - dolfin.Constant(dolfin.assemble(param["k"] * Fs[0].dx(0) * IC_s.dx(0) * dx_s(i)) \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2]))) \
                    * var_F * Bx[i] * Bx[i] * IC_r * det_J[i] * dolfin.dx(meshes[1]) 
            if nE > 0:
                for old in range(nE):
                    for j in range(0,3):
                        l +=- dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                            * var_F * PGD_func[1][old] * det_J[j] * Bt[j][0][1] * dolfin.dx(meshes[1]) \
                            - dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[j][1] * PGD_func[0][old] * dx_s(j)) \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                            * var_F * PGD_func[1][old].dx(0) * det_J[j] * dolfin.dx(meshes[1]) \
                            - dolfin.Constant(dolfin.assemble(param["k"] * Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                            * var_F * Bx[j] * Bx[j] * PGD_func[1][old] * det_J[j] * dolfin.dx(meshes[1])
                            
    if typ == 'Q':
            l =  dolfin.Constant( dolfin.assemble(Fs[0] * Q[0][0] * dx_s(1)) \
                * dolfin.assemble(Fs[1] * Q[1][0] * det_J[1] * dolfin.dx(meshes[1]))) \
                * var_F * Q[2][0] * dolfin.dx(meshes[2])
            for i in range(0,3):
                l +=- dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][0][0] * IC_s.dx(0) * dx_s(i)) \
                    * dolfin.assemble(Fs[1] * IC_r * det_J[i] * Bt[i][0][1] * dolfin.dx(meshes[1]))) \
                    * var_F * IC_Q * dolfin.dx(meshes[2]) \
                    - dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][1] * IC_s * dx_s(i)) \
                    * dolfin.assemble(Fs[1] * IC_r.dx(0) * det_J[i] * dolfin.dx(meshes[1]))) \
                    * var_F * IC_Q * dolfin.dx(meshes[2]) \
                    - dolfin.Constant(dolfin.assemble(param["k"] * Fs[0].dx(0) * IC_s.dx(0) * dx_s(i)) \
                    * dolfin.assemble(Fs[1] * Bx[i] * Bx[i] * IC_r * det_J[i] * dolfin.dx(meshes[1]))) \
                    * var_F * IC_Q * dolfin.dx(meshes[2])
            if nE > 0:
                for old in range(nE):
                    for j in range(0,3):
                        l +=- dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                            * dolfin.assemble(Fs[1] * PGD_func[1][old] * det_J[j] * Bt[j][0][1] * dolfin.dx(meshes[1]))) \
                            * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                            - dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[j][1] * PGD_func[0][old] * dx_s(j)) \
                            * dolfin.assemble(Fs[1] * PGD_func[1][old].dx(0) * det_J[j] * dolfin.dx(meshes[1]))) \
                            * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                            - dolfin.Constant(dolfin.assemble(param["k"] * Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                            * dolfin.assemble(Fs[1] * Bx[j] * Bx[j] * PGD_func[1][old] * det_J[j] * dolfin.dx(meshes[1]))) \
                            * var_F * PGD_func[2][old] * dolfin.dx(meshes[2])
                            
    return l

def problem_assemble_lhs_FD(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem
    
    dx_s = dolfin.Measure('dx', domain=meshes[0], subdomain_data=dom[0])

    # define expressions
    Bt = param["Bt"] 
    Bx = param["Bx"] 
    det_J = param["det_J"] 
            
    a = 0
    if typ == 's':
        for i in range(3):
            alpha_t1 = Fs[1].vector()[:].transpose() @ param['M1'] @ (Fs[1].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:])
            alpha_t2 = (Fs[1].vector()[:] * Bt[i][1]).transpose() @ param['D1_up'] @ Fs[1].vector()[:] 
            alpha_t3 = Fs[1].vector()[:].transpose() @ param['M1'] @ (Fs[1].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:])
                        
            a +=  dolfin.Constant(alpha_t1 \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
                * param["rho"] * param["c_p"] * var_F * Bt[i][0][0] * fct_F.dx(0) * dx_s(i) \
                + dolfin.Constant(alpha_t2 \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
                * param["rho"] * param["c_p"] * var_F * fct_F * dx_s(i) \
                + dolfin.Constant(alpha_t3 \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))) \
                * param["k"] * var_F.dx(0) * fct_F.dx(0) * dx_s(i)
                
    if typ == 'r':
        for i in range(3): 
            a +=  dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][0][0] * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * param['M1'] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:] \
                + dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * Bt[i][1] * param['D1_up'] \
                + dolfin.assemble(param["k"] * Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * param['M1'] * Bx[i].vector()[:] * Bx[i].vector()[:] * det_J[i].vector()[:]
        
        # add initial condition
        a[:,param['bc_idx']] = 0
        a[param['bc_idx'],:] = 0
        a[param['bc_idx'], param['bc_idx']] = 1
                
    if typ == 'Q':
        for i in range(3):
            alpha_t1 = Fs[1].vector()[:].transpose() @ param['M1'] @ (Fs[1].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:])
            alpha_t2 = (Fs[1].vector()[:] * Bt[i][1]).transpose() @ param['D1_up'] @ Fs[1].vector()[:]
            alpha_t3 = Fs[1].vector()[:].transpose() @ param['M1'] @ (Fs[1].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:])
            
            a +=  dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][0][0] * Fs[0].dx(0) * dx_s(i)) \
                * alpha_t1 ) \
                * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Fs[0] * dx_s(i)) \
                * alpha_t2 ) \
                * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(param["k"] * Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * alpha_t3 ) \
                * var_F * fct_F * dolfin.dx(meshes[2])

    return a

def problem_assemble_rhs_FD(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem
    
    dx_s = dolfin.Measure('dx', domain=meshes[0], subdomain_data=dom[0])

    # define expressions
    Bt = param["Bt"] 
    Bx = param["Bx"] 
    det_J = param["det_J"]
    IC_s = param["IC_s"] 
    IC_r = param["IC_r"] 
    IC_Q = param["IC_Q"]

    beta_t1 = Fs[1].vector()[:].transpose() @ param['M1'] @ (Q[1][0].vector()[:] * det_J[1].vector()[:])
    
    if typ == 's':
            l =   dolfin.Constant(beta_t1 \
                * dolfin.assemble(Fs[2] * Q[2][0] * dolfin.dx(meshes[2]))) \
                * var_F * Q[0][0] * dx_s(1)
            for i in range(0,3):
                beta_t2 = Fs[1].vector()[:].transpose() @ param['M1'] @ (IC_r.vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:])
                beta_t3 = (Fs[1].vector()[:] * Bt[i][1]).transpose() @ param['D1_up'] @ IC_r.vector()[:]
                beta_t4 = Fs[1].vector()[:].transpose() @ param['M1'] @ (IC_r.vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:])
                
                l +=- dolfin.Constant(beta_t2 \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["c_p"] * var_F * Bt[i][0][0] * IC_s.dx(0) * dx_s(i) \
                    - dolfin.Constant(beta_t3 \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2]))) \
                    * param["rho"] * param["c_p"] * var_F * IC_s * dx_s(i) \
                    - dolfin.Constant(beta_t4 \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2]))) \
                    * param["k"] * var_F.dx(0) * IC_s.dx(0) * dx_s(i) 
            if nE > 0:
                for old in range(nE):
                    for j in range(0,3):
                        alpha_t1 = Fs[1].vector()[:].transpose() @ param['M1'] @ (PGD_func[1][old].vector()[:] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:])
                        alpha_t2 = (Fs[1].vector()[:] * Bt[j][1]).transpose() @ param['D1_up'] @ PGD_func[1][old].vector()[:]
                        alpha_t3 = Fs[1].vector()[:].transpose() @ param['M1'] @ (PGD_func[1][old].vector()[:] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:])
                        
                        l +=- dolfin.Constant(alpha_t1 \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                            * param["rho"] * param["c_p"] * var_F * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j) \
                            - dolfin.Constant( alpha_t2 \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                            * param["rho"] * param["c_p"] * var_F * PGD_func[0][old] * dx_s(j) \
                            - dolfin.Constant(alpha_t3 \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))) \
                            * param["k"] * var_F.dx(0) * PGD_func[0][old].dx(0) * dx_s(j) 
                            
    if typ == 'r':
            l =   dolfin.assemble(Fs[0] * Q[0][0] * dx_s(1)) \
                * dolfin.assemble(Fs[2] * Q[2][0] * dolfin.dx(meshes[2])) \
                * param['M1'] @ Q[1][0].vector()[:] * det_J[1].vector()[:]
            for i in range(0,3):
                l +=- dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][0][0] * IC_s.dx(0) * dx_s(i)) \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2])) \
                    * param['M1'] @ IC_r.vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:] \
                    - dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * IC_s * dx_s(i)) \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2])) \
                    * Bt[i][1] * param['D1_up'] @ IC_r.vector()[:] \
                    - dolfin.assemble(param["k"] * Fs[0].dx(0) * IC_s.dx(0) * dx_s(i)) \
                    * dolfin.assemble(Fs[2] * IC_Q * dolfin.dx(meshes[2])) \
                    * param['M1'] @ IC_r.vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]
            if nE > 0:
                for old in range(nE):
                    for j in range(0,3):
                        l +=- dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                            * param['M1'] @ PGD_func[1][old].vector()[:] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:] \
                            - dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * PGD_func[0][old] * dx_s(j)) \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                            * Bt[j][1] * param['D1_up'] @ PGD_func[1][old].vector()[:] \
                            - dolfin.assemble(param["k"] * Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                            * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                            * param['M1'] @ PGD_func[1][old].vector()[:] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:]
                            
            # add initial condition
            l[param['bc_idx']] = 0
                            
    if typ == 'Q':
            l =  dolfin.Constant(dolfin.assemble(Fs[0] * Q[0][0] * dx_s(1)) \
                * beta_t1 ) \
                * var_F * Q[2][0] * dolfin.dx(meshes[2])
            for i in range(0,3):
                beta_t2 = Fs[1].vector()[:].transpose() @ param['M1'] @ (IC_r.vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:])
                beta_t3 = (Fs[1].vector()[:] * Bt[i][1]).transpose() @ param['D1_up'] @ IC_r.vector()[:]
                beta_t4 = Fs[1].vector()[:].transpose() @ param['M1'] @ (IC_r.vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:])
                
                l +=- dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[i][0][0] * IC_s.dx(0) * dx_s(i)) \
                    * beta_t2 ) \
                    * var_F * IC_Q * dolfin.dx(meshes[2]) \
                    - dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * IC_s * dx_s(i)) \
                    * beta_t3 ) \
                    * var_F * IC_Q * dolfin.dx(meshes[2]) \
                    - dolfin.Constant(dolfin.assemble(param["k"] * Fs[0].dx(0) * IC_s.dx(0) * dx_s(i)) \
                    * beta_t4 ) \
                    * var_F * IC_Q * dolfin.dx(meshes[2])
            if nE > 0:
                for old in range(nE):
                    for j in range(0,3):
                        alpha_t1 = Fs[1].vector()[:].transpose() @ param['M1'] @ (PGD_func[1][old].vector()[:] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:])
                        alpha_t2 = (Fs[1].vector()[:] * Bt[j][1]).transpose() @ param['D1_up'] @ PGD_func[1][old].vector()[:]
                        alpha_t3 = Fs[1].vector()[:].transpose() @ param['M1'] @ (PGD_func[1][old].vector()[:] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:])
                        
                        l +=- dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                            * alpha_t1 ) \
                            * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                            - dolfin.Constant(dolfin.assemble(param["rho"] * param["c_p"] * Fs[0] * PGD_func[0][old] * dx_s(j)) \
                            * alpha_t2 ) \
                            * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                            - dolfin.Constant(dolfin.assemble(param["k"] * Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                            * alpha_t3 ) \
                            * var_F * PGD_func[2][old] * dolfin.dx(meshes[2])

    return l

def main_FEM(vs, param):
    '''computation of PGD solution for given problem normal'''
    
    # define position functions
    param.update({"h_g": param["L"]/20})
    param.update({"h_1": dolfin.interpolate(dolfin.Expression('x[0]-(h_g/2)', degree=4, h_g=param["h_g"]),vs[1])})
    param.update({"h_2": dolfin.interpolate(dolfin.Expression('L-(h_g/2)-x[0]', degree=4, h_g=param["h_g"], L=param["L"]),vs[1])})
    
    # define derivates with respect to s/r and jacobian
    Bt = [[[dolfin.Expression('-vel*x[0]', degree=2, vel=param["vel"]), dolfin.Expression('1/h_1', degree=2, h_1=param["h_1"])], 
               dolfin.Constant(param["vel"])], 
          [[dolfin.Constant(-param["vel"]), dolfin.Constant(1/param["h_g"])], 
               dolfin.Constant(param["vel"])], 
          [[dolfin.Expression('vel*(x[0]-2)-vel', degree=2, vel=param["vel"]), dolfin.Expression('1/h_2', degree=2, h_2=param["h_2"])], 
               dolfin.Constant(param["vel"])]]    
    Bx = [1/param["h_1"], 1/param["h_g"], 1/param["h_2"]] 
    det_J = [param["h_1"]/param["vel"], param["h_g"]/param["vel"], param["h_2"]/param["vel"]]
    param.update({'Bt': Bt})
    param.update({'Bx': Bx})
    param.update({'det_J': det_J})
    
    # define nonhomogeneous dirichlet IC in s and r
    param.update({'IC_s': dolfin.interpolate(dolfin.Expression('1.0', degree=4),vs[0])})
    param.update({'IC_r': dolfin.interpolate(dolfin.Expression('(x[0] <= r_0 + 1E-8) ? Tamb : 0', degree=4, r_0=param["r_0"], Tamb=param["T_amb"]),vs[1])})
    param.update({'IC_Q': dolfin.interpolate(dolfin.Expression('1.0', degree=4),vs[2])})
    
    # define heat source in s, r and Q 
    qs = [dolfin.interpolate(dolfin.Expression('exp(-3*pow((x[0]-1)*h_g-af,2)/pow(af,2))', degree=4, af=param["af"], h_g=param["h_g"]),vs[0])]
    qr = [dolfin.interpolate(dolfin.Expression('6*sqrt(3) / ((af+af)*af*af*pow(pi,3/2))', degree=4, af=param["af"]), vs[1])]
    qQ = [dolfin.interpolate(dolfin.Expression('x[0]', degree=4),vs[2])]

    prob = ['s', 'r', 'Q'] # problems according problem_assemble_fcts
    seq_fp = np.arange(len(vs)) # default sequence of Fixed Point iteration according to prob
    PGD_nmax = 10       # max number of PGD modes

    pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-sr', name_coord=['s', 'r', 'Q'],
                           modes_info=['T_x', 'Node', 'Scalar'],
                           Vs=vs, dom_fct=create_dom, bc_fct=create_bc, load=[qs,qr,qQ],
                           param=param, rhs_fct=problem_assemble_rhs_FEM,
                           lhs_fct=problem_assemble_lhs_FEM, probs=prob, seq_fp=seq_fp,
                           PGD_nmax=PGD_nmax)
    
    pgd_prob.max_fp_it = 20
    pgd_prob.tol_fp_it = 1e-5
    # pgd_prob.stop_fp = 'norm'
    pgd_prob.stop_fp = 'chady'
    pgd_prob.fp_init = 'randomized'
    
    pgd_prob.solve_PGD(_problem='linear')
    print(pgd_prob.simulation_info)
    print('PGD Amplitude', pgd_prob.amplitude)

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance

    return pgd_s, param

def main_FD(vs, param):
    '''computation of PGD solution for given problem normal'''
    
    # define position functions
    param.update({"h_g": param["L"]/20})
    param.update({"h_1": dolfin.interpolate(dolfin.Expression('x[0]-(h_g/2)', degree=4, h_g=param["h_g"]),vs[1])})
    param.update({"h_2": dolfin.interpolate(dolfin.Expression('L-(h_g/2)-x[0]', degree=4, h_g=param["h_g"], L=param["L"]),vs[1])})
    
    # define derivates with respect to s/r and jacobian
    Bt = [[[dolfin.Expression('-vel*x[0]', degree=2, vel=param["vel"]), dolfin.interpolate(dolfin.Expression('1/h_1', degree=2, h_1=param["h_1"]), vs[1])], 
               param["vel"]], 
          [[dolfin.Constant(-param["vel"]), dolfin.interpolate(dolfin.Constant(1/param["h_g"]), vs[1])], 
               param["vel"]], 
          [[dolfin.Expression('vel*(x[0]-2)-vel', degree=2, vel=param["vel"]), dolfin.interpolate(dolfin.Expression('1/h_2', degree=2, h_2=param["h_2"]), vs[1])], 
               param["vel"]]]    
    Bx = [dolfin.interpolate(dolfin.Expression('1/h_1', degree=2, h_1=param["h_1"]), vs[1]), 
          dolfin.interpolate(dolfin.Expression('1/h_g', degree=2, h_g=param["h_g"]), vs[1]),
          dolfin.interpolate(dolfin.Expression('1/h_2', degree=2, h_2=param["h_2"]), vs[1])]
    det_J = [dolfin.interpolate(dolfin.Expression('h_1/v', degree=2, h_1=param["h_1"], v=param["vel"]), vs[1]),
             dolfin.interpolate(dolfin.Expression('h_g/v', degree=2, h_g=param["h_g"], v=param["vel"]), vs[1]),
             dolfin.interpolate(dolfin.Expression('h_2/v', degree=2, h_2=param["h_2"], v=param["vel"]), vs[1])]
    param.update({'Bt': Bt})
    param.update({'Bx': Bx})
    param.update({'det_J': det_J})
    
    # define nonhomogeneous dirichlet IC in s and r
    param.update({'IC_s': dolfin.interpolate(dolfin.Expression('1.0', degree=4),vs[0])})
    param.update({'IC_r': dolfin.interpolate(dolfin.Expression('(x[0] <= r_0 + 1E-8) ? Tamb : 0', degree=4, r_0=param["r_0"], Tamb=param["T_amb"]),vs[1])})
    param.update({'IC_Q': dolfin.interpolate(dolfin.Expression('1.0', degree=4),vs[2])})
    
    # define heat source in s, r and Q 
    qs = [dolfin.interpolate(dolfin.Expression('exp(-3*pow((x[0]-1)*h_g-af,2)/pow(af,2))', degree=4, af=param["af"], h_g=param["h_g"]),vs[0])]
    qr = [dolfin.interpolate(dolfin.Expression('6*sqrt(3) / ((af+af)*af*af*pow(pi,3/2))', degree=4, af=param["af"]), vs[1])]
    qQ = [dolfin.interpolate(dolfin.Expression('x[0]', degree=4),vs[2])]

    prob = ['s', 'r', 'Q'] # problems according problem_assemble_fcts
    seq_fp = np.arange(len(vs)) # default sequence of Fixed Point iteration according to prob
    PGD_nmax = 10       # max number of PGD modes

    pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-sr', name_coord=['s', 'r', 'Q'],
                           modes_info=['T_x', 'Node', 'Scalar'],
                           Vs=vs, dom_fct=create_dom, bc_fct=create_bc, load=[qs,qr,qQ],
                           param=param, rhs_fct=problem_assemble_rhs_FD,
                           lhs_fct=problem_assemble_lhs_FD, probs=prob, seq_fp=seq_fp,
                           PGD_nmax=PGD_nmax)
    
    pgd_prob.max_fp_it = 20
    pgd_prob.tol_fp_it = 1e-5
    # pgd_prob.stop_fp = 'norm'
    pgd_prob.stop_fp = 'chady'
    pgd_prob.fp_init = 'randomized'
    
    pgd_prob.solve_PGD(_problem='linear', solve_modes=["FEM","FD","FEM"])
    print(pgd_prob.simulation_info)
    print('PGD Amplitude', pgd_prob.amplitude)

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance

    return pgd_s, param

def remapping(pgd_solution, param, data_test):
    # compute remapping of heating phase
    
    # initialize data
    r_mesh = pgd_solution.mesh[1].dataX
    time = r_mesh/param["vel"]
    x_0 = data_test[0]
    Q = data_test[1]
    
    # map back to x in PGD
    PGD_heating = np.zeros(len(r_mesh))
    for i, rr in enumerate(r_mesh):
        u_pgd = pgd_solution.evaluate(0, [1,2], [rr,Q], 0)

        # left side
        s_temp = x_0 / param["h_1"](rr)
        if s_temp < 1:
            PGD_heating[i] = u_pgd(s_temp)
        else:
            # center
            s_temp = (x_0 - param["h_1"](rr)) / param["h_g"] + 1
            if s_temp <= 2:
                PGD_heating[i] = u_pgd(s_temp)
            else:
                # right side
                s_temp = (x_0 - param["h_1"](rr) - param["h_g"]) / param["h_2"](rr) + 2
                PGD_heating[i] = u_pgd(s_temp)
    
    # add initial condition to heating phase
    PGD_heating[0] = param["T_amb"]
    
    return PGD_heating, time

# TEST: PGD result VS FEM result
#==============================================================================  

# Finite Element Model
#=======================================
class Reference_solution():
    
    def __init__(self,Vs=[], param=[], meshes=[]):
        
        self.Vs = Vs # Location
        self.param = param # Parameters
        self.meshes = meshes # Meshes
        
        self.meshes_ref, self.Vs_ref = create_meshes([500,400], [1,1], [[0, self.param["L"]], [0,10]])
                
    def fem_definition(self,x_fixed, Q): 
        # initialize parameter
        rho = self.param['rho']                                 # material density [kg/m³]
        k = self.param['k']                                     # heat conductivity [W/m°C]
        cp = self.param['c_p']                                  # specific heat capacity [J/kg°C]
        time_points = self.meshes[1].coordinates()/self.param["vel"]
        dt = time_points[1][0]-time_points[0][0]

        # Define initial value
        T_n = fenics.Function(self.Vs_ref[0])
        T_n.vector()[:] = np.ones(len(T_n.vector()[:]))*self.param["T_amb"]
        
        # Define goldak heat input         
        q = fenics.Expression('6*sqrt(3)*Q / ((af+af)*af*af*pow(pi,3/2)) * exp(-3*pow(x[0]-vel*t,2)/pow(af,2))\
                              ', degree=2, af=self.param["af"], h_g=self.param["h_g"], Q=Q, vel=self.param["vel"], t=0)
        
        # Define problem functions
        T = fenics.TrialFunction(self.Vs_ref[0])
        v = fenics.TestFunction(self.Vs_ref[0])

        # Collect variational form         
        F =  rho*cp*T*v*fenics.dx \
            + dt*k*fenics.dot(fenics.grad(T), fenics.grad(v))*fenics.dx \
            - (dt*q + rho*cp*T_n)*v*fenics.dx
        a, L = fenics.lhs(F), fenics.rhs(F)
        
        # Time-stepping
        T = fenics.Function(self.Vs_ref[0])
        T_time = [self.param["T_amb"]]
        for n in range(len(time_points)-1):
            t = time_points[n][0]
            q.t=t
            
            # Compute solution
            fenics.solve(a == L, T)
            # Update previous solution
            T_n.assign(T)
            
            T_time.append(T(x_fixed))
        
        return np.array(T_time)
                    
    def __call__(self, data_test):
        
        # sampled variable values
        x_fixed = data_test[0]    # fixed position
        Q = data_test[1]          # heat input
                
        ref_sol = self.fem_definition(x_fixed, Q)
        
        return ref_sol
    
# PGD model and Error computation
#=======================================
class PGDproblem(unittest.TestCase):

    def setUp(self):
        
        # global parameters
        self.ord = 1  # order for each mesh
        self.ords = [self.ord, self.ord, self.ord]
        self.ranges = [[0., 3.],     # smin, smax
                  [0.0035, 0.0965],  # rmin, rmax
                  [900, 1100]]      # Qmin, Qmax   
        self.num_elem = [500, # number of elements in s
                         400, # number of elements in r
                         100] # number of elements in Q
        
        # sampling parameters
        self.fixed_dim = [0] # Fixed variable
        self.n_samples = 10 # Number of samples

        # define some parameters
        self.param = {"rho": 7100, "c_p": 3100, "k": 100, "r_0": self.ranges[1][0], "vel": 0.01, "L": 0.1, "af": 0.002, "T_amb": 25}

    def TearDown(self):
        pass

    def test_solver(self):
        
        # MESH
        #======================================================================
        meshes, Vs = create_meshes(self.num_elem, self.ords, self.ranges)
        
        # create FD matrices for time mesh
        # sort dof coordinates and create FD matrices
        x_dofs = np.array(Vs[1].tabulate_dof_coordinates()[:].flatten())
        idx_sort = np.argsort(x_dofs)
        Mt, _, D1_upt = FD_matrices(Vs[1].tabulate_dof_coordinates()[idx_sort])
        # idx for initial condition of time problem
        bc_idx = np.where(x_dofs == self.param["r_0"])

        # store re_sorted according dofs!
        self.param['M1'] = Mt[idx_sort,:][:,idx_sort]
        self.param['D1_up'] = D1_upt[idx_sort, :][:, idx_sort]
        self.param['bc_idx'] = bc_idx[0]
        
        data_test = [[0.05, 1000]] # x, Q
        
        # Computing solution with FD and error
        #======================================================================
        pgd_test_FEM, param_FEM = main_FEM(Vs, self.param) # Computing PGD with FEM
        pgd_test_FD, param_FD = main_FD(Vs, self.param) # Computing PGD with FD        

        fun_FOM = Reference_solution(Vs=Vs, param=param_FEM, meshes=meshes) # Computing Full-Order model: FEM
        
        # error_uPGD = PGDErrorComputation(fixed_dim = self.fixed_dim,
        #                                   n_samples = self.n_samples,
        #                                   FOM_model = fun_FOM,
        #                                   PGD_model = pgd_test
        #                                   )
        
        # errorL2, mean_errorL2, max_errorL2 = error_uPGD.evaluate_error() # Computing Error
        
        # print('Mean error',mean_errorL2)
        # print('Max. error',max_errorL2)
        
        # self.assertTrue(mean_errorL2<0.004)
                
        # compute reference solution
        u_fem = fun_FOM(data_test[0])
        # remap PGD to plot usefull information
        remap_FEM, time_FEM = remapping(pgd_solution=pgd_test_FEM, param=param_FEM, data_test=data_test[0])
        remap_FD, time_FD = remapping(pgd_solution=pgd_test_FD, param=param_FD, data_test=data_test[0])
        
        plt.figure()
        plt.plot(time_FD, remap_FD, '-*b', label="PGD FD")
        plt.plot(time_FEM, remap_FEM, '-*g', label="PGD FEM")
        plt.plot(time_FEM, u_fem, '-or', label="FEM")
        plt.title(f"PGD solution at {data_test[0][0]}m over time")
        plt.xlabel("Time s [s]")
        plt.ylabel("Temperature T [°C]")
        plt.legend()
        plt.draw()
        plt.show()
        
        
        
if __name__ == '__main__':
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging
    logging.basicConfig(level=logging.INFO)

    unittest.main()