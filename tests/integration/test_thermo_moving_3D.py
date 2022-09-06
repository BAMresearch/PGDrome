'''
    3D PGD example (heat equation with a moving goldak heat source) with five PGD variables (space (x,y,z), time and heat input)
    solving PGD problem using FD non space and FEM space
    returning PGDModel (as forward model) or PGD instance

'''

import unittest
import dolfin
import numpy as np

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

    return [0, 0, 0, initCond, 0] # s, y, z, r, eta

def problem_assemble_lhs(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem
    
    dx_s = dolfin.Measure('dx', domain=meshes[0], subdomain_data=dom[0])

    # define expressions
    Bt = param["Bt"] 
    Bx = param["Bx"] 
    det_J = param["det_J"] 
            
    alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ Fs[4].vector()[:]
                    
    a = 0
    if typ == 's':
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ Fs[3].vector()[:] 
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
                        
            a +=  dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r1 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * Bt[i][0][0] * fct_F.dx(0) * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * fct_F * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * alpha_eta1) \
                * param["k"] * var_F.dx(0) * fct_F.dx(0) * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F * fct_F * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F * fct_F * dx_s(i)
                    
    if typ == 'y':
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ Fs[3].vector()[:] 
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
                        
            a +=  dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0].dx(0) * Bt[i][0][0] * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r1 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * alpha_eta1) \
                * param["k"] * var_F * fct_F * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F.dx(0) * fct_F.dx(0) * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F * fct_F * dolfin.dx(meshes[1])
    
    if typ == 'z':
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ Fs[3].vector()[:] 
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
                        
            a +=  dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0].dx(0) * Bt[i][0][0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * alpha_r1 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * alpha_r2 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * alpha_r3 \
                * alpha_eta1) \
                * param["k"] * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F.dx(0) * fct_F.dx(0) * dolfin.dx(meshes[2])    
                    
    if typ == 'r':
        for i in range(3): 
            a +=  dolfin.assemble(Fs[0] * Bt[i][0][0] * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["rho"] * param["c_p"] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:] * param['M_r'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["rho"] * param["c_p"] * Bt[i][1] * param['D1_up_r'] \
                + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["k"] * Bx[i].vector()[:] * Bx[i].vector()[:] * det_J[i].vector()[:] * param['M_r'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["k"] * det_J[i].vector()[:] * param['M_r'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["k"] * det_J[i].vector()[:] * param['M_r']
        
        # add initial condition
        a[:,param['bc_idx']] = 0
        a[param['bc_idx'],:] = 0
        a[param['bc_idx'], param['bc_idx']] = 1
                
    if typ == 'eta':
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ Fs[3].vector()[:] 
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            
            a +=  dolfin.assemble(Fs[0] * Bt[i][0][0] * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r1 \
                * param["rho"] * param["c_p"] * param['M_eta'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * param["rho"] * param["c_p"] * param['M_eta'] \
                + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * param["k"] * param['M_eta'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * param["k"] * param['M_eta'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * param["k"] * param['M_eta']

    return a

def problem_assemble_rhs(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem
    
    dx_s = dolfin.Measure('dx', domain=meshes[0], subdomain_data=dom[0])

    # define expressions
    Bt = param["Bt"] 
    Bx = param["Bx"] 
    det_J = param["det_J"]
    IC = [param["IC_s"], param["IC_y"], param["IC_z"], param["IC_r"], param["IC_eta"]]

    beta_r1 = (Fs[3].vector()[:] * det_J[1].vector()[:]).transpose() @ param['M_r'] @ Q[3].vector()[:]
    beta_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ Q[4].vector()[:]
    alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ IC[4].vector()[:]
    
    if typ == 's':
        l =   dolfin.Constant(dolfin.assemble(Fs[1] * Q[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Q[2] * dolfin.dx(meshes[2])) \
            * beta_r1 \
            * beta_eta1) \
            * var_F * Q[0] * dx_s(1)
        for i in range(0,3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ IC[3].vector()[:] 
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
                
            l +=- dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r1 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * Bt[i][0][0] * IC[0].dx(0) * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * IC[0] * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * alpha_eta1) \
                * param["k"] * var_F.dx(0) * IC[0].dx(0) * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F * IC[0] * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F * IC[0] * dx_s(i)                     
        if nE > 0:
            for old in range(nE):
                alpha_eta2 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                for j in range(0,3):
                    alpha_old_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ PGD_func[3][old].vector()[:] 
                    alpha_old_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                        
                    l +=- dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r1 \
                        * alpha_eta2) \
                        * param["rho"] * param["c_p"] * var_F * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r2 \
                        * alpha_eta2) \
                        * param["rho"] * param["c_p"] * var_F * PGD_func[0][old] * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r3 \
                        * alpha_eta2) \
                        * param["k"] * var_F.dx(0) * PGD_func[0][old].dx(0) * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_eta2) \
                        * param["k"] * var_F * PGD_func[0][old] * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_eta2) \
                        * param["k"] * var_F * PGD_func[0][old] * dx_s(j)
                        
    if typ == 'y':
        l =   dolfin.Constant(dolfin.assemble(Fs[0] * Q[0] * dx_s(1)) \
            * dolfin.assemble(Fs[2] * Q[2] * dolfin.dx(meshes[2])) \
            * beta_r1 \
            * beta_eta1) \
            * var_F * Q[1] * dolfin.dx(meshes[1])
        for i in range(0,3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ IC[3].vector()[:] 
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
                
            l +=- dolfin.Constant(dolfin.assemble(Fs[0] * Bt[i][0][0] * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r1 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * IC[1] * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * IC[1] * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * alpha_eta1) \
                * param["k"] * var_F * IC[1] * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F.dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[2].dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F * IC[1] * dolfin.dx(meshes[1])
        if nE > 0:
            for old in range(nE):
                alpha_eta2 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                for j in range(0,3):
                    alpha_old_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ PGD_func[3][old].vector()[:] 
                    alpha_old_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                        
                    l +=- dolfin.Constant(dolfin.assemble(Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r1 \
                        * alpha_eta2) \
                        * param["rho"] * param["c_p"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r2 \
                        * alpha_eta2) \
                        * param["rho"] * param["c_p"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r3 \
                        * alpha_eta2) \
                        * param["k"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_eta2) \
                        * param["k"] * var_F.dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_eta2) \
                        * param["k"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1])
                        
    if typ == 'z':
        l =   dolfin.Constant(dolfin.assemble(Fs[0] * Q[0] * dx_s(1)) \
            * dolfin.assemble(Fs[1] * Q[1] * dolfin.dx(meshes[1])) \
            * beta_r1 \
            * beta_eta1) \
            * var_F * Q[2] * dolfin.dx(meshes[2])
        for i in range(0,3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ IC[3].vector()[:] 
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
                
            l +=- dolfin.Constant(dolfin.assemble(Fs[0] * Bt[i][0][0] * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * alpha_r1 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * alpha_r2 \
                * alpha_eta1) \
                * param["rho"] * param["c_p"] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * alpha_r3 \
                * alpha_eta1) \
                * param["k"] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * alpha_r4 \
                * alpha_eta1) \
                * param["k"] * var_F.dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2])
        if nE > 0:
            for old in range(nE):
                alpha_eta2 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                for j in range(0,3):
                    alpha_old_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ PGD_func[3][old].vector()[:] 
                    alpha_old_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                        
                    l +=- dolfin.Constant(dolfin.assemble(Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * alpha_old_r1 \
                        * alpha_eta2) \
                        * param["rho"] * param["c_p"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * alpha_old_r2 \
                        * alpha_eta2) \
                        * param["rho"] * param["c_p"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * alpha_old_r3 \
                        * alpha_eta2) \
                        * param["k"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                        * alpha_old_r4 \
                        * alpha_eta2) \
                        * param["k"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * alpha_old_r4 \
                        * alpha_eta2) \
                        * param["k"] * var_F.dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])
                        
    if typ == 'r':
        l =   dolfin.assemble(Fs[0] * Q[0] * dx_s(1)) \
            * dolfin.assemble(Fs[1] * Q[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Q[2] * dolfin.dx(meshes[2])) \
            * beta_eta1 \
            * param['M_r'] @ Q[3].vector()[:] * det_J[1].vector()[:]
        for i in range(0,3):
            l +=- dolfin.assemble(Fs[0] * Bt[i][0][0] * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["rho"] * param["c_p"] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:] * param['M_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["rho"] * param["c_p"] * Bt[i][1] * param['D1_up_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0].dx(0) * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["k"] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:] * param['M_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["k"] * det_J[i].vector()[:] * param['M_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * param["k"] * det_J[i].vector()[:] * param['M_r'] @ IC[3].vector()[:]
        if nE > 0:
            for old in range(nE):
                alpha_eta2 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                for j in range(0,3):
                    l +=- dolfin.assemble(Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_eta2 \
                        * param["rho"] * param["c_p"] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_eta2 \
                        * param["rho"] * param["c_p"] * Bt[j][1] * param['D1_up_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_eta2 \
                        * param["k"] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_eta2 \
                        * param["k"] * det_J[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                        * alpha_eta2 \
                        * param["k"] * det_J[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:]
                            
        # add initial condition
        l[param['bc_idx']] = 0
                            
    if typ == 'eta':
        l =  dolfin.assemble(Fs[0] * Q[0] * dx_s(1)) \
            * dolfin.assemble(Fs[1] * Q[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Q[2] * dolfin.dx(meshes[2])) \
            * beta_r1 \
            * param['M_eta'] @ Q[4].vector()[:]
        for i in range(0,3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ IC[3].vector()[:] 
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
                
            l +=- dolfin.assemble(Fs[0] * Bt[i][0][0] * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r1 \
                * param["rho"] * param["c_p"] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * param["rho"] * param["c_p"] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0].dx(0) * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * param["k"] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * param["k"] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * param["k"] * param['M_eta'] @ IC[4].vector()[:]
        if nE > 0:
            for old in range(nE):
                for j in range(0,3):
                    alpha_old_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ PGD_func[3][old].vector()[:] 
                    alpha_old_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                        
                    l +=- dolfin.assemble(Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r1 \
                        * param["rho"] * param["c_p"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r2 \
                        * param["rho"] * param["c_p"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r3 \
                        * param["k"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * param["k"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * param["k"] * param['M_eta'] @ PGD_func[4][old].vector()[:]

    return l

def create_PGD(param={}, vs=[], q_PGD=None, q_coeff=None):
    
    # define position functions
    param.update({"h_1": dolfin.interpolate(dolfin.Expression('x[0]-(h_g/2)', degree=4, h_g=param["h_g"]),vs[3])})
    param.update({"h_2": dolfin.interpolate(dolfin.Expression('L-(h_g/2)-x[0]', degree=4, h_g=param["h_g"], L=param["L"]),vs[3])})
    
    # define derivates with respect to s/r and jacobian
    Bt = [[[dolfin.Expression('-vel*x[0]', degree=2, vel=param["vel"]), dolfin.interpolate(dolfin.Expression('1/h_1', degree=2, h_1=param["h_1"]), vs[3])], 
               param["vel"]], 
          [[dolfin.Constant(-param["vel"]), dolfin.interpolate(dolfin.Constant(1/param["h_g"]), vs[3])], 
               param["vel"]], 
          [[dolfin.Expression('vel*(x[0]-2)-vel', degree=2, vel=param["vel"]), dolfin.interpolate(dolfin.Expression('1/h_2', degree=2, h_2=param["h_2"]), vs[13])], 
               param["vel"]]]    
    Bx = [dolfin.interpolate(dolfin.Expression('1/h_1', degree=2, h_1=param["h_1"]), vs[3]), 
          dolfin.interpolate(dolfin.Expression('1/h_g', degree=2, h_g=param["h_g"]), vs[3]),
          dolfin.interpolate(dolfin.Expression('1/h_2', degree=2, h_2=param["h_2"]), vs[3])]
    det_J = [dolfin.interpolate(dolfin.Expression('h_1/v', degree=2, h_1=param["h_1"], v=param["vel"]), vs[3]),
             dolfin.interpolate(dolfin.Expression('h_g/v', degree=2, h_g=param["h_g"], v=param["vel"]), vs[3]),
             dolfin.interpolate(dolfin.Expression('h_2/v', degree=2, h_2=param["h_2"], v=param["vel"]), vs[3])]
    param.update({'Bt': Bt})
    param.update({'Bx': Bx})
    param.update({'det_J': det_J})
    
    # define nonhomogeneous dirichlet IC
    param.update({'IC_s': dolfin.interpolate(param['IC_s'],vs[0])})
    param.update({'IC_y': dolfin.interpolate(param['IC_y'],vs[1])})
    param.update({'IC_z': dolfin.interpolate(param['IC_z'],vs[2])})
    param.update({'IC_r': dolfin.interpolate(param['IC_r'],vs[3])})
    param.update({'IC_eta': dolfin.interpolate(param['IC_eta'],vs[4])})
    
    # define heat source in s, y, z, r and eta
    q_s = dolfin.interpolate(q_PGD, vs[0])
    q_y = dolfin.interpolate(dolfin.Constant(1.0),vs[1])
    q_z = dolfin.interpolate(dolfin.Constant(1.0),vs[2])
    q_r = dolfin.interpolate(dolfin.Expression('q_coeff', q_coeff=q_coeff, degree=1),vs[3])
    q_eta = dolfin.interpolate(dolfin.Expression('x[0]*P', P=param['P'], degree=1), vs[4])
    
    # create FD matrices from meshes
    r_dofs = np.array(vs[3].tabulate_dof_coordinates()[:].flatten())
    r_sort = np.argsort(r_dofs)
    M_r, _, D1_up_r = FD_matrices(r_dofs[r_sort])
    param['M_r'],param['D1_up_r'] = M_r[r_sort,:][:,r_sort], D1_up_r[r_sort,:][:,r_sort]
    param['bc_idx']=np.where(r_dofs==0)[0]
    
    eta_dofs = np.array(vs[4].tabulate_dof_coordinates()[:].flatten())
    eta_sort = np.argsort(eta_dofs)
    M_eta, _, _ = FD_matrices(eta_dofs[eta_sort])
    param['M_eta'] = M_eta[eta_sort,:][:,eta_sort]
    
    solve_modes = ["FEM", "FEM", "FEM", "FD", "FD"]
        
    pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-SYZREta', name_coord=['S', 'Y', 'Z', 'R', 'Eta'],
                           modes_info=['T', 'Node', 'Scalar'],
                           Vs=vs, dom_fct=create_dom, bc_fct=create_bc, load=[q_s, q_y, q_z, q_r, q_eta],
                           param=param, rhs_fct=problem_assemble_rhs, lhs_fct=problem_assemble_lhs,
                           probs=['s', 'y', 'z', 'r', 'eta'], seq_fp=np.arange(len(vs)),
                           PGD_nmax=20, PGD_tol=1e-5)    
                           
    pgd_prob.MM = [0, 0, 0, param['M_r'], param['M_eta']]  # for norms!

    pgd_prob.stop_fp = 'norm'
    pgd_prob.max_fp_it = 20
    pgd_prob.tol_fp_it = 1e-5
    pgd_prob.norm_modes = 'stiff'
    
    pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes)

    print(pgd_prob.simulation_info)
    print('PGD Amplitude', pgd_prob.amplitude)

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance
    
    return pgd_s, param

def remapping(pgd_solution, param, pos_fixed=None, eta_fixed=None):
    # compute remapping of heating phase
    
    # initialize data
    r_mesh = pgd_solution.mesh[1].dataX
    time = r_mesh/param["vel"]
    
    x_fixed = pos_fixed[0]
    y_fixed = pos_fixed[1]
    z_fixed = pos_fixed[2]
    
    # map back to x in PGD
    PGD_heating = np.zeros(len(r_mesh))
    for i, rr in enumerate(r_mesh):
        u_pgd = pgd_solution.evaluate(0, [1,2,3,4], [y_fixed,z_fixed,rr,eta_fixed], 0)

        # left side
        s_temp = x_fixed / param["h_1"](rr)
        if s_temp < 1:
            PGD_heating[i] = u_pgd(s_temp)
        else:
            # center
            s_temp = (x_fixed - param["h_1"](rr)) / param["h_g"] + 1
            if s_temp <= 2:
                PGD_heating[i] = u_pgd(s_temp)
            else:
                # right side
                s_temp = (x_fixed - param["h_1"](rr) - param["h_g"]) / param["h_2"](rr) + 2
                PGD_heating[i] = u_pgd(s_temp)
    
    # add initial condition to heating phase
    # PGD_heating[0] = param["T_amb"]
    PGD_heating += param['IC_x'](x_fixed) * param['IC_y'](y_fixed) * param['IC_z'](z_fixed) \
        * param['IC_r'].compute_vertex_values()[:] * param["IC_eta"](eta_fixed)
    
    return PGD_heating, time

# TEST: PGD result VS FEM result
#==============================================================================  

# Finite Element Model
#=======================================
class Reference():
    # TODO -> ab hier weiter anpassen!!!
    def __init__(self, param={}, vs=[], q=None, x_fixed=None):
        
        self.vs = vs # Location
        self.param = param # Parameters
        self.q = q # source term

        # time points
        self.time_mesh = self.vs[1].mesh().coordinates()[:]
        self.T_n = dolfin.interpolate(self.param["Tamb_fct"], self.vs[0])

        # problem
        self.mesh = self.vs[0].mesh()
        T = dolfin.TrialFunction(self.vs[0])
        v = dolfin.TestFunction(self.vs[0])
        self.dt = dolfin.Constant(1.)
        self.Q = dolfin.Constant(1.)
        self.F = self.param["rho"] * self.param["c_p"] * T * v * dolfin.dx() \
            + self.dt * self.param["k"] * dolfin.dot(dolfin.grad(T), dolfin.grad(v)) * dolfin.dx() \
            - (self.dt * self.Q * self.q + self.param["rho"] * self.param["c_p"] * self.T_n) * v * dolfin.dx()

        self.fixed_x = x_fixed
        
    def __call__(self, values):

        # check time mesh for requested time value
        if np.where(self.time_mesh == values[0])[0] == []:
            print("ERROR time step not in mesh What to do?")
        self.Q.assign(values[1]*self.param["P"])
        
        # Time-stepping
        Ttime = []
        Ttmp = dolfin.Function(self.vs[0])
        Ttmp.vector()[:] = 1 * self.T_n.vector()[:]
        Ttime.append(Ttmp) # otherwise it will be overwritten with new solution
        Txfixed = [np.copy(self.T_n(self.fixed_x))]
        T = dolfin.Function(self.vs[0])
        for i in range(len(self.time_mesh)-1):
            self.dt.assign(self.time_mesh[i+1][0]-self.time_mesh[i][0])
            (self.q).t = self.time_mesh[i][0]
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
            
        return Ttime, Txfixed # solution in time over x and time solution at fixed x
    
            
# PGD model and Error computation
#=======================================
class problem(unittest.TestCase):

    def setUp(self):
        
        # define some parameters
        self.param = {"rho": 7100, "c_p": 3100, "k": 100, "r_0": 0.01, "vel": 0.01, "L": 0.1, "af": 0.005, "T_amb": 25, "P": 1000, "t_max": 10}
        self.param.update({"h_g": self.param["af"]*3})
        
        # global parameters
        self.ord = 1  # order for each mesh
        self.ords = [self.ord, self.ord, self.ord]
        self.ranges_PGD = [[0., 3.],                                                                # smin, smax
                  [self.param["r_0"], self.param["vel"]*self.param["t_max"]-self.param["r_0"]],     # rmin, rmax
                  [0.5, 1.]]                                                                        # etamin, etamax   
        self.ranges_FEM = [[0., self.param["L"]],                                                                   # xmin, xmax
                  [self.param["r_0"]/self.param["vel"], self.param["t_max"]-self.param["r_0"]/self.param["vel"]],   # tmin, tmax
                  [0.5, 1.]]                                                                                        # etamin, etamax   
        self.num_elem = [500, # number of elements in x, s
                         400, # number of elements in t, r
                         100] # number of elements in eta
        
        # evaluation parameters
        self.t_fixed = 0.9*self.param['t_max'] 
        self.r_fixed = self.t_fixed * self.param["vel"]
        self.eta_fixed = 1.
        self.x_fixed = 0.5*self.param['L'] 

        self.plotting = True
        # self.plotting = False

    def TearDown(self):
        pass
                          
    def test_heating(self):
        # case heating
        self.q_coeff = 6 * np.sqrt(3) / ((self.param["af"]+self.param["af"]) * self.param["af"] * self.param["af"] * np.pi**(3/2))
        self.q_FEM = dolfin.Expression('x[0] >= vel*t+hg/2 ? 0 : x[0] <= vel*t-hg/2 ? 0 : \
                                       coeff * exp(-3*(pow(x[0]-vel*t,2)/pow(af,2)))', \
                                    degree=4, coeff=self.q_coeff, af=self.param['af'], vel=self.param["vel"], hg=self.param["h_g"], t=0)
        self.q_PGD = dolfin.Expression('exp(-3*pow((x[0]-1.5)*h_g,2)/pow(af,2))', degree=4, af=self.param["af"], h_g=self.param["h_g"])

        self.param['Tamb_fct'] = dolfin.Expression('Tamb', degree=1, Tamb=self.param["T_amb"]) #initial condition FEM
        self.param['IC_t'] = self.param['Tamb_fct']
        self.param['IC_x'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_eta'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_r'] = self.param['Tamb_fct']
        self.param['IC_s'] = dolfin.Expression('1.0', degree=1)
        
        # MESH
        meshes_FEM, vs_FEM = create_meshes(self.num_elem, self.ords, self.ranges_FEM)
        meshes_PGD, vs_PGD = create_meshes(self.num_elem, self.ords, self.ranges_PGD)
        
        # PGD
        pgd_fd, param = create_PGD(param=self.param, vs=vs_PGD, q_PGD=self.q_PGD, q_coeff=self.q_coeff)
        
        # FEM reference solution 
        u_fem, u_fem2 = Reference(param=self.param, vs=vs_FEM, q=self.q_FEM, x_fixed=self.x_fixed)(
            [self.t_fixed, self.eta_fixed])

        # PGD solution at fixed place over time
        upgd_fd2, time_PGD = remapping(pgd_fd, self.param, self.x_fixed, self.eta_fixed)
        
        # error computation
        errors_FEM22 = np.linalg.norm(upgd_fd2 - u_fem2) / np.linalg.norm(u_fem2)  # PGD FD - FEM
        print('error in time FD', errors_FEM22)

        if self.plotting:
            #### plotting optional
            import matplotlib.pyplot as plt
            
            plt.figure()
            plt.plot(meshes_FEM[1].coordinates()[:], u_fem2, '-or', label='FEM')
            plt.plot(time_PGD, upgd_fd2, '-*g', label='PGD FD')
            plt.title(f"PGD solution at [x,eta]={self.x_fixed},{self.eta_fixed} over time")
            plt.xlabel("time t [s]")
            plt.ylabel("Temperature T [°C]")
            plt.legend()

            plt.draw()
            plt.show()

        # self.assertTrue(errors_FEM21 < 1e-2) 
        self.assertTrue(errors_FEM22 < 1e-2) 

if __name__ == '__main__':
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging
    logging.basicConfig(level=logging.INFO)

    unittest.main()