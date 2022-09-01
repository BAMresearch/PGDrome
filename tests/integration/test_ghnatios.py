'''
    compare pgdrome module with matlab code from Ghnatios
    "attempt_1_x_y_q_u0.m"

    problem:    k(\partial^2 T/\partial x^2 + \partial^2 T/\partial y²) + Q = 0
                T(x_0) = (1-1/3x) u0 for x_0 in [0, lX]
                Q(x,y,q,u0) = (1 if x<L/2 \\ 0 if x>L/2) * q

    PGD approach: T=sum F(x)F(y)F(q)F(u0)

'''

import unittest
import dolfin as df
import numpy as np

from pgdrome.solver import PGDProblem1, FD_matrices
from pgdrome.model import PGDErrorComputation

def create_meshes(num_elem, ord, ranges):
    #create 1D meshes per variable

    meshes = list()
    Vs = list()

    dim = len(num_elem)

    for i in range(dim):
        mesh_tmp = df.IntervalMesh(num_elem[i], ranges[i][0], ranges[i][1])
        Vs_tmp = df.FunctionSpace(mesh_tmp, 'CG', ord[i])

        meshes.append(mesh_tmp)
        Vs.append(Vs_tmp)

    return meshes, Vs

def get_coordinates_and_sorts(vs):
    # get dof coordinates from 1D meshes and sorting for creating FD matrices

    x_dofs = list()
    idx_sort = list()

    for i in range(len(vs)):
        x_dofs.append(np.array(vs[i].tabulate_dof_coordinates()[:].flatten()))
        idx_sort.append(np.argsort(x_dofs[i]))

    return x_dofs, idx_sort

def create_bc(Vs,dom,param):
    # boundary conditions list

    def leftright(x, on_boundary):
        return on_boundary and df.near(x[0], 0.0, 1e-6) or df.near(x[0], param["lx"], 1e-6)

    Cond = df.DirichletBC(Vs[0], 0, leftright) #homogenous boundary

    return [Cond, 0, 0, 0] #x,y,q,u0

def problem_assemble_lhs_FEM(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem

    if typ == 'r':
        a = df.Constant(df.assemble(Fs[1] * Fs[1] * df.dx(meshes[1])) \
            * df.assemble(Fs[2] * Fs[2] * df.dx(meshes[2])) \
            * df.assemble(Fs[3] * Fs[3] * df.dx(meshes[3])) ) \
            * param["k"] * fct_F.dx(0) * var_F.dx(0) * df.dx(meshes[0]) \
            + df.Constant(df.assemble(Fs[1].dx(0) * Fs[1].dx(0) * df.dx(meshes[1])) \
            * df.assemble(Fs[2] * Fs[2] * df.dx(meshes[2])) \
            * df.assemble(Fs[3] * Fs[3] * df.dx(meshes[3])) ) \
            * param["k"] * fct_F * var_F * df.dx(meshes[0]) # k*dT/dx*DT/dx + k*dT/dx*DT/dx
    if typ == 's':
        a = df.Constant(df.assemble(Fs[0].dx(0) * Fs[0].dx(0) * df.dx(meshes[0])) \
            * df.assemble(Fs[2] * Fs[2] * df.dx(meshes[2])) \
            * df.assemble(Fs[3] * Fs[3] * df.dx(meshes[3])) ) \
            * param["k"] * fct_F * var_F * df.dx(meshes[1]) \
            + df.Constant(df.assemble(Fs[0] * Fs[0] * df.dx(meshes[0])) \
            * df.assemble(Fs[2] * Fs[2] * df.dx(meshes[2])) \
            * df.assemble(Fs[3] * Fs[3] * df.dx(meshes[3])) ) \
            * param["k"] * fct_F.dx(0) * var_F.dx(0) * df.dx(meshes[1]) # k*dT/dx*DT/dx + k*dT/dx*DT/dx
    if typ == 't':
        a = df.Constant(df.assemble(Fs[0].dx(0) * Fs[0].dx(0) * df.dx(meshes[0])) \
            * df.assemble(Fs[1] * Fs[1] * df.dx(meshes[1])) \
            * df.assemble(Fs[3] * Fs[3] * df.dx(meshes[3])) ) \
            * param["k"] * fct_F * var_F * df.dx(meshes[2]) \
            + df.Constant(df.assemble(Fs[0] * Fs[0] * df.dx(meshes[0])) \
            * df.assemble(Fs[1].dx(0) * Fs[1].dx(0) * df.dx(meshes[1])) \
            * df.assemble(Fs[3] * Fs[3] * df.dx(meshes[3])) ) \
            * param["k"] * fct_F * var_F * df.dx(meshes[2]) # k*dT/dx*DT/dx + k*dT/dx*DT/dx
    if typ == 'u':
        a = df.Constant(df.assemble(Fs[0].dx(0) * Fs[0].dx(0) * df.dx(meshes[0])) \
            * df.assemble(Fs[1] * Fs[1] * df.dx(meshes[1])) \
            * df.assemble(Fs[2] * Fs[2] * df.dx(meshes[2])) ) \
            * param["k"] * fct_F * var_F * df.dx(meshes[3]) \
            + df.Constant(df.assemble(Fs[0] * Fs[0] * df.dx(meshes[0])) \
            * df.assemble(Fs[1].dx(0) * Fs[1].dx(0) * df.dx(meshes[1])) \
            * df.assemble(Fs[2] * Fs[2] * df.dx(meshes[2])) ) \
            * param["k"] * fct_F * var_F * df.dx(meshes[3]) # k*dT/dx*DT/dx + k*dT/dx*DT/dx

    return a

def problem_assemble_rhs_FEM(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem

    BC_x = param["BC_x"]
    BC_y = param["BC_y"]
    BC_q = param["BC_q"]
    BC_u = param["BC_u0"]

    if typ == 'r':
        l = df.Constant(df.assemble(Q[1][0] * Fs[1] * df.dx(meshes[1])) \
            * df.assemble(Q[2][0] * Fs[2] * df.dx(meshes[2])) \
            * df.assemble(Q[3][0] * Fs[3] * df.dx(meshes[3])) ) \
            * Q[0][0] * var_F * df.dx(meshes[0]) #\
#            - df.Constant(df.assemble(BC_y * Fs[1] * df.dx(meshes[1])) \
#            * df.assemble(BC_q * Fs[2] * df.dx(meshes[2])) \
#            * df.assemble(BC_u * Fs[3] * df.dx(meshes[3])) ) \
#            * param["k"] * BC_x.dx(0) * var_F.dx(0) * df.dx(meshes[0]) \
#            - df.Constant(df.assemble(BC_y.dx(0) * Fs[1].dx(0) * df.dx(meshes[1])) \
#            * df.assemble(BC_q * Fs[2] * df.dx(meshes[2])) \
#            * df.assemble(BC_u * Fs[3] * df.dx(meshes[3])) ) \
#            * param["k"] * BC_x * var_F * df.dx(meshes[0]) # T*Q - (k*dT/dx*DTBC/dx + k*dT/dx*DTBC/dx)

        if nE > 0:
            for old in range(nE):
                l +=- df.Constant(param["alpha"][old] * df.assemble(PGD_func[1][old] * Fs[1] * df.dx(meshes[1])) \
                    * df.assemble(PGD_func[2][old] * Fs[2] * df.dx(meshes[2])) \
                    * df.assemble(PGD_func[3][old] * Fs[3] * df.dx(meshes[3])) ) \
                    * param["k"] * PGD_func[0][old].dx(0) * var_F.dx(0) * df.dx(meshes[0]) \
                    - df.Constant(param["alpha"][old] * df.assemble(PGD_func[1][old].dx(0) * Fs[1].dx(0) * df.dx(meshes[1])) \
                    * df.assemble(PGD_func[2][old]  * Fs[2] * df.dx(meshes[2])) \
                    * df.assemble(PGD_func[3][old]  * Fs[3] * df.dx(meshes[3])) ) \
                    * param["k"] * PGD_func[0][old]  * var_F * df.dx(meshes[0]) # -k*dT/dx*DTold/dx - k*dT/dx*DTold/dx

    if typ == 's':
        l = df.Constant(df.assemble(Q[0][0] * Fs[0] * df.dx(meshes[0])) \
            * df.assemble(Q[2][0] * Fs[2] * df.dx(meshes[2])) \
            * df.assemble(Q[3][0] * Fs[3] * df.dx(meshes[3])) ) \
            * Q[1][0] * var_F * df.dx(meshes[1]) #\
#            - df.Constant(df.assemble(BC_x.dx(0) * Fs[0].dx(0) * df.dx(meshes[0])) \
#            * df.assemble(BC_q * Fs[2] * df.dx(meshes[2])) \
#            * df.assemble(BC_u * Fs[3] * df.dx(meshes[3])) ) \
#            * param["k"] * BC_y * var_F * df.dx(meshes[1]) \
#            - df.Constant(df.assemble(BC_x * Fs[0] * df.dx(meshes[0])) \
#            * df.assemble(BC_q * Fs[2] * df.dx(meshes[2])) \
#            * df.assemble(BC_u * Fs[3] * df.dx(meshes[3])) ) \
#            * param["k"] * BC_y.dx(0) * var_F.dx(0) * df.dx(meshes[1]) # T*Q - (k*dT/dx*DTBC/dx + k*dT/dx*DTBC/dx)

        if nE > 0:
            for old in range(nE):
                l +=- df.Constant(param["alpha"][old]* df.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * df.dx(meshes[0])) \
                    * df.assemble(PGD_func[2][old] * Fs[2] * df.dx(meshes[2])) \
                    * df.assemble(PGD_func[3][old] * Fs[3] * df.dx(meshes[3])) ) \
                    * param["k"] * PGD_func[1][old] * var_F * df.dx(meshes[1]) \
                    - df.Constant(param["alpha"][old]*df.assemble(PGD_func[0][old] * Fs[0] * df.dx(meshes[0])) \
                    * df.assemble(PGD_func[2][old]  * Fs[2] * df.dx(meshes[2])) \
                    * df.assemble(PGD_func[3][old]  * Fs[3] * df.dx(meshes[3])) ) \
                    * param["k"] * PGD_func[1][old].dx(0)  * var_F.dx(0) * df.dx(meshes[1]) # -k*dT/dx*DTold/dx - k*dT/dx*DTold/dx

    if typ == 't':
        l = df.Constant(df.assemble(Q[0][0] * Fs[0] * df.dx(meshes[0])) \
            * df.assemble(Q[1][0] * Fs[1] * df.dx(meshes[1])) \
            * df.assemble(Q[3][0] * Fs[3] * df.dx(meshes[3])) ) \
            * Q[2][0] * var_F * df.dx(meshes[2]) #\
#            - df.Constant(df.assemble(BC_x.dx(0) * Fs[0].dx(0) * df.dx(meshes[0])) \
#            * df.assemble(BC_y * Fs[1] * df.dx(meshes[1])) \
#            * df.assemble(BC_u * Fs[3] * df.dx(meshes[3])) ) \
#            * param["k"] * BC_q * var_F * df.dx(meshes[2]) \
#            - df.Constant(df.assemble(BC_x * Fs[0] * df.dx(meshes[0])) \
#            * df.assemble(BC_y.dx(0) * Fs[1].dx(0) * df.dx(meshes[1])) \
#            * df.assemble(BC_u * Fs[3] * df.dx(meshes[3])) ) \
#            * param["k"] * BC_q * var_F * df.dx(meshes[2]) # T*Q - (k*dT/dx*DTBC/dx + k*dT/dx*DTBC/dx)

        if nE > 0:
            for old in range(nE):
                l +=- df.Constant(param["alpha"][old]* df.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * df.dx(meshes[0])) \
                    * df.assemble(PGD_func[1][old] * Fs[1] * df.dx(meshes[1])) \
                    * df.assemble(PGD_func[3][old] * Fs[3] * df.dx(meshes[3])) ) \
                    * param["k"] * PGD_func[2][old] * var_F * df.dx(meshes[2]) \
                    - df.Constant(param["alpha"][old]* df.assemble(PGD_func[0][old] * Fs[0] * df.dx(meshes[0])) \
                    * df.assemble(PGD_func[1][old].dx(0) * Fs[1].dx(0) * df.dx(meshes[1])) \
                    * df.assemble(PGD_func[3][old] * Fs[3] * df.dx(meshes[3])) ) \
                    * param["k"] * PGD_func[2][old] * var_F * df.dx(meshes[2]) # -k*dT/dx*DTold/dx - k*dT/dx*DTold/dx

    if typ == 'u':
        l = df.Constant(df.assemble(Q[0][0] * Fs[0] * df.dx(meshes[0])) \
            * df.assemble(Q[1][0] * Fs[1] * df.dx(meshes[1])) \
            * df.assemble(Q[2][0] * Fs[2] * df.dx(meshes[2])) ) \
            * Q[3][0] * var_F * df.dx(meshes[3]) #\
#            - df.Constant(df.assemble(BC_x.dx(0) * Fs[0].dx(0) * df.dx(meshes[0])) \
#            * df.assemble(BC_y * Fs[1] * df.dx(meshes[1])) \
#            * df.assemble(BC_q * Fs[2] * df.dx(meshes[2])) ) \
#            * param["k"] * BC_u * var_F * df.dx(meshes[3]) \
#            - df.Constant(df.assemble(BC_x * Fs[0] * df.dx(meshes[0])) \
#            * df.assemble(BC_y.dx(0) * Fs[1].dx(0) * df.dx(meshes[1])) \
#            * df.assemble(BC_q * Fs[2] * df.dx(meshes[2])) ) \
#            * param["k"] * BC_u * var_F * df.dx(meshes[3]) # T*Q - (k*dT/dx*DTBC/dx + k*dT/dx*DTBC/dx)

        if nE > 0:
            for old in range(nE):
                l +=- df.Constant(param["alpha"][old]* df.assemble(PGD_func[0][old].dx(0) * Fs[0].dx(0) * df.dx(meshes[0])) \
                    * df.assemble(PGD_func[1][old] * Fs[1] * df.dx(meshes[1])) \
                    * df.assemble(PGD_func[2][old] * Fs[2] * df.dx(meshes[2])) ) \
                    * param["k"] * PGD_func[3][old] * var_F * df.dx(meshes[3]) \
                    - df.Constant(param["alpha"][old]* df.assemble(PGD_func[0][old] * Fs[0] * df.dx(meshes[0])) \
                    * df.assemble(PGD_func[1][old].dx(0) * Fs[1].dx(0) * df.dx(meshes[1])) \
                    * df.assemble(PGD_func[2][old] * Fs[2] * df.dx(meshes[2])) ) \
                    * param["k"] * PGD_func[3][old] * var_F * df.dx(meshes[3]) # -k*dT/dx*DTold/dx - k*dT/dx*DTold/dx

    return l

def problem_assemble_lhs_FD(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem discription left hand side of DGL for each fixed point problem

    if typ == 'r':
        a = -Fs[1].vector()[:].transpose() @ param['M_y'] @ Fs[1].vector()[:] \
            * Fs[2].vector()[:].transpose() @ param['M_q'] @ Fs[2].vector()[:] \
            * Fs[3].vector()[:].transpose() @ param['M_u'] @ Fs[3].vector()[:] \
            * param["k"] * param['D2_x'] \
            - Fs[1].vector()[:].transpose() @ param['D2_y'] @ Fs[1].vector()[:] \
            * Fs[2].vector()[:].transpose() @ param['M_q'] @ Fs[2].vector()[:] \
            * Fs[3].vector()[:].transpose() @ param['M_u'] @ Fs[3].vector()[:] \
            * param["k"] * param['M_x'] # -k*dT/dx*DT/dx - k*dT/dx*DT/dx
        #boundary condition only in x!
        # add initial condition
        a[:, param['bc_idx']] = 0.
        a[param['bc_idx'], :] = 0.
        a[param['bc_idx'], param['bc_idx']] = 1.0

    if typ == 's':
        a = -Fs[0].vector()[:].transpose() @ param['D2_x'] @ Fs[0].vector()[:] \
            * Fs[2].vector()[:].transpose() @ param['M_q'] @ Fs[2].vector()[:] \
            * Fs[3].vector()[:].transpose() @ param['M_u'] @ Fs[3].vector()[:] \
            * param["k"] * param['M_y'] \
            - Fs[0].vector()[:].transpose() @ param['M_x'] @ Fs[0].vector()[:] \
            * Fs[2].vector()[:].transpose() @ param['M_q'] @ Fs[2].vector()[:] \
            * Fs[3].vector()[:].transpose() @ param['M_u'] @ Fs[3].vector()[:] \
            * param["k"] * param['D2_y'] # -k*dT/dx*DT/dx - k*dT/dx*DT/dx
    if typ == 't':
        a = -Fs[0].vector()[:].transpose() @ param['D2_x'] @ Fs[0].vector()[:] \
            * Fs[1].vector()[:].transpose() @ param['M_y'] @ Fs[1].vector()[:] \
            * Fs[3].vector()[:].transpose() @ param['M_u'] @ Fs[3].vector()[:] \
            * param["k"] * param['M_q'] \
            - Fs[0].vector()[:].transpose() @ param['M_x'] @ Fs[0].vector()[:] \
            * Fs[1].vector()[:].transpose() @ param['D2_y'] @ Fs[1].vector()[:] \
            * Fs[3].vector()[:].transpose() @ param['M_u'] @ Fs[3].vector()[:] \
            * param["k"] * param['M_q'] # -k*dT/dx*DT/dx - k*dT/dx*DT/dx
    if typ == 'u':
        a = -Fs[0].vector()[:].transpose() @ param['D2_x'] @ Fs[0].vector()[:] \
            * Fs[1].vector()[:].transpose() @ param['M_y'] @ Fs[1].vector()[:] \
            * Fs[2].vector()[:].transpose() @ param['M_q'] @ Fs[2].vector()[:] \
            * param["k"] * param['M_u'] \
            - Fs[0].vector()[:].transpose() @ param['M_x'] @ Fs[0].vector()[:] \
            * Fs[1].vector()[:].transpose() @ param['D2_y'] @ Fs[1].vector()[:] \
            * Fs[2].vector()[:].transpose() @ param['M_q'] @ Fs[2].vector()[:] \
            * param["k"] * param['M_u'] # -k*dT/dx*DT/dx - k*dT/dx*DT/dx

    return a

def problem_assemble_rhs_FD(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem discription right hand side of DGL for each fixed point problem

    BC_x = param["BC_x"]
    BC_y = param["BC_y"]
    BC_q = param["BC_q"]
    BC_u = param["BC_u0"]

    if typ == 'r':
        l = Fs[1].vector()[:].transpose() @ param['M_y'] @ Q[1][0].vector()[:].transpose() \
            * Fs[2].vector()[:].transpose() @ param['M_q'] @ Q[2][0].vector()[:].transpose() \
            * Fs[3].vector()[:].transpose() @ param['M_u'] @ Q[3][0].vector()[:].transpose()  \
            * param['M_x'] @ Q[0][0].vector()[:] #\
            # + Fs[1].vector()[:].transpose() @ param['M_y'] @ BC_y.vector()[:] \
            # * Fs[2].vector()[:].transpose() @ param['M_q'] @ BC_q.vector()[:] \
            # * Fs[3].vector()[:].transpose() @ param['M_u'] @ BC_u.vector()[:] \
            # * param["k"] * param['D2_x'] @ BC_x.vector()[:]  \
            # + Fs[1].vector()[:].transpose() @ param['D2_y'] @ BC_y.vector()[:] \
            # * Fs[2].vector()[:].transpose() @ param['M_q'] @ BC_q.vector()[:] \
            # * Fs[3].vector()[:].transpose() @ param['M_u'] @ BC_u.vector()[:] \
            # * param["k"] * param['M_x'] @ BC_x.vector()[:] # T*Q - -(k*dT/dx*DTBC/dx + k*dT/dx*DTBC/dx)

        if nE > 0:
            for old in range(nE):
                l +=+ param["alpha"][old]*Fs[1].vector()[:].transpose() @ param['M_y'] @ PGD_func[1][old].vector()[:] \
                    * Fs[2].vector()[:].transpose() @ param['M_q'] @ PGD_func[2][old].vector()[:] \
                    * Fs[3].vector()[:].transpose() @ param['M_u'] @ PGD_func[3][old].vector()[:] \
                    * param["k"] * param['D2_x'] @ PGD_func[0][old].vector()[:] \
                    + param["alpha"][old]*Fs[1].vector()[:].transpose() @ param['D2_y'] @ PGD_func[1][old].vector()[:] \
                    * Fs[2].vector()[:].transpose() @ param['M_q'] @ PGD_func[2][old].vector()[:] \
                    * Fs[3].vector()[:].transpose() @ param['M_u'] @ PGD_func[3][old].vector()[:] \
                    * param["k"] * param['M_x'] @ PGD_func[0][old].vector()[:] # - (-k*dT/dx*DTold/dx - k*dT/dx*DTold/dx)

        # add initial condition
        l[param['bc_idx']] = 0

    if typ == 's':
        l = Fs[0].vector()[:].transpose() @ param['M_x'] @ Q[0][0].vector()[:].transpose() \
            * Fs[2].vector()[:].transpose() @ param['M_q'] @ Q[2][0].vector()[:].transpose() \
            * Fs[3].vector()[:].transpose() @ param['M_u'] @ Q[3][0].vector()[:].transpose()  \
            * param['M_y'] @ Q[1][0].vector()[:] #\
            # + Fs[0].vector()[:].transpose() @ param['D2_x'] @ BC_x.vector()[:] \
            # * Fs[2].vector()[:].transpose() @ param['M_q'] @ BC_q.vector()[:] \
            # * Fs[3].vector()[:].transpose() @ param['M_u'] @ BC_u.vector()[:] \
            # * param["k"] * param['M_y'] @ BC_y.vector()[:] \
            # + Fs[0].vector()[:].transpose() @ param['M_x'] @ BC_x.vector()[:] \
            # * Fs[2].vector()[:].transpose() @ param['M_q'] @ BC_q.vector()[:] \
            # * Fs[3].vector()[:].transpose() @ param['M_u'] @ BC_u.vector()[:] \
            # * param["k"] * param['D2_y'] @ BC_y.vector()[:]  # T*Q - -(k*dT/dx*DTBC/dx + k*dT/dx*DTBC/dx)

        if nE > 0:
            for old in range(nE):
                l +=+ param["alpha"][old]*Fs[0].vector()[:].transpose() @ param['D2_x'] @ PGD_func[0][old].vector()[:] \
                    * Fs[2].vector()[:].transpose() @ param['M_q'] @ PGD_func[2][old].vector()[:] \
                    * Fs[3].vector()[:].transpose() @ param['M_u'] @ PGD_func[3][old].vector()[:] \
                    * param["k"] * param['M_y'] @ PGD_func[1][old].vector()[:] \
                    + param["alpha"][old]*Fs[0].vector()[:].transpose() @ param['M_x'] @ PGD_func[0][old].vector()[:] \
                    * Fs[2].vector()[:].transpose() @ param['M_q'] @ PGD_func[2][old].vector()[:] \
                    * Fs[3].vector()[:].transpose() @ param['M_u'] @ PGD_func[3][old].vector()[:] \
                    * param["k"] * param['D2_y'] @ PGD_func[1][old].vector()[:] # -(-k*dT/dx*DTold/dx - k*dT/dx*DTold/dx)

    if typ == 't':
        l = Fs[0].vector()[:].transpose() @ param['M_x'] @ Q[0][0].vector()[:].transpose() \
            * Fs[1].vector()[:].transpose() @ param['M_y'] @ Q[1][0].vector()[:].transpose() \
            * Fs[3].vector()[:].transpose() @ param['M_u'] @ Q[3][0].vector()[:].transpose()  \
            * param['M_q'] @ Q[2][0].vector()[:] #\
            # + Fs[0].vector()[:].transpose() @ param['D2_x'] @ BC_x.vector()[:] \
            # * Fs[1].vector()[:].transpose() @ param['M_y'] @ BC_y.vector()[:] \
            # * Fs[3].vector()[:].transpose() @ param['M_u'] @ BC_u.vector()[:] \
            # * param["k"] * param['M_q'] @ BC_q.vector()[:] \
            # + Fs[0].vector()[:].transpose() @ param['M_x'] @ BC_x.vector()[:] \
            # * Fs[1].vector()[:].transpose() @ param['D2_y'] @ BC_y.vector()[:] \
            # * Fs[3].vector()[:].transpose() @ param['M_u'] @ BC_u.vector()[:] \
            # * param["k"] * param['M_q'] @ BC_q.vector()[:]  # T*Q - -(k*dT/dx*DTBC/dx + k*dT/dx*DTBC/dx)

        if nE > 0:
            for old in range(nE):
                l +=+ param["alpha"][old]*Fs[0].vector()[:].transpose() @ param['D2_x'] @ PGD_func[0][old].vector()[:] \
                    * Fs[1].vector()[:].transpose() @ param['M_y'] @ PGD_func[1][old].vector()[:] \
                    * Fs[3].vector()[:].transpose() @ param['M_u'] @ PGD_func[3][old].vector()[:] \
                    * param["k"] * param['M_q'] @ PGD_func[2][old].vector()[:] \
                    + param["alpha"][old]*Fs[0].vector()[:].transpose() @ param['M_x'] @ PGD_func[0][old].vector()[:] \
                    * Fs[1].vector()[:].transpose() @ param['D2_y'] @ PGD_func[1][old].vector()[:] \
                    * Fs[3].vector()[:].transpose() @ param['M_u'] @ PGD_func[3][old].vector()[:] \
                    * param["k"] * param['M_q'] @ PGD_func[2][old].vector()[:] # -(-k*dT/dx*DTold/dx - k*dT/dx*DTold/dx)

    if typ == 'u':
        l = Fs[0].vector()[:].transpose() @ param['M_x'] @ Q[0][0].vector()[:].transpose() \
            * Fs[1].vector()[:].transpose() @ param['M_y'] @ Q[1][0].vector()[:].transpose() \
            * Fs[2].vector()[:].transpose() @ param['M_q'] @ Q[2][0].vector()[:].transpose()  \
            * param['M_u'] @ Q[3][0].vector()[:] #\
            # + Fs[0].vector()[:].transpose() @ param['D2_x'] @ BC_x.vector()[:] \
            # * Fs[1].vector()[:].transpose() @ param['M_y'] @ BC_y.vector()[:] \
            # * Fs[2].vector()[:].transpose() @ param['M_q'] @ BC_q.vector()[:] \
            # * param["k"] * param['M_u'] @ BC_u.vector()[:] \
            # + Fs[0].vector()[:].transpose() @ param['M_x'] @ BC_x.vector()[:] \
            # * Fs[1].vector()[:].transpose() @ param['D2_y'] @ BC_y.vector()[:] \
            # * Fs[2].vector()[:].transpose() @ param['M_q'] @ BC_q.vector()[:] \
            # * param["k"] * param['M_u'] @ BC_u.vector()[:]  # T*Q - -(k*dT/dx*DTBC/dx + k*dT/dx*DTBC/dx)

        if nE > 0:
            for old in range(nE):
                l +=+ param["alpha"][old]*Fs[0].vector()[:].transpose() @ param['D2_x'] @ PGD_func[0][old].vector()[:] \
                    * Fs[1].vector()[:].transpose() @ param['M_y'] @ PGD_func[1][old].vector()[:] \
                    * Fs[2].vector()[:].transpose() @ param['M_q'] @ PGD_func[2][old].vector()[:] \
                    * param["k"] * param['M_u'] @ PGD_func[3][old].vector()[:] \
                    + param["alpha"][old]*Fs[0].vector()[:].transpose() @ param['M_x'] @ PGD_func[0][old].vector()[:] \
                    * Fs[1].vector()[:].transpose() @ param['D2_y'] @ PGD_func[1][old].vector()[:] \
                    * Fs[2].vector()[:].transpose() @ param['M_q'] @ PGD_func[2][old].vector()[:] \
                    * param["k"] * param['M_u'] @ PGD_func[3][old].vector()[:] # -(-k*dT/dx*DTold/dx - k*dT/dx*DTold/dx)

    return l

def create_PGD(param=[], vs=[], _type=None):

        #define nonhomgeneous BC
        # param['BC_x'] = df.interpolate(df.Expression('x[0]<0+1e-8 ? 1.0-1.0/3.0*x[0] : (x[0]>L-1e-8 ? 1.0-1.0/3.0*x[0] : 0)',degree=1, L=param['lx']),vs[0])
        param['BC_x'] = df.interpolate(df.Expression('1.0-1.0/3.0*x[0]',degree=1),vs[0])
        param['BC_y'] = df.interpolate(df.Expression('1.0',degree=1),vs[1])
        param['BC_q'] = df.interpolate(df.Expression('1.0',degree=1),vs[2])
        param['BC_u0'] = df.interpolate(df.Expression('x[0]',degree=1),vs[3])

        #define source term as interpolated for FD version!
        qx=[df.interpolate(df.Expression('x[0]<L/2 ? 1.0 : 0', degree=1, L=param['lx']),vs[0])]
        qy=[df.interpolate(df.Expression('1.0',degree=1),vs[1])]
        qq=[df.interpolate(df.Expression('x[0]',degree=1),vs[2])]
        qu0=[df.interpolate(df.Expression('1.0',degree=1),vs[3])]

        #create PGD problem
        if _type == 'FEM':
            ass_rhs = problem_assemble_rhs_FEM
            ass_lhs = problem_assemble_lhs_FEM
            solve_modes = ["FEM","FEM","FEM","FEM"]

        elif _type == 'FD':
            # create FD matrices from meshes
            x_dofs, idx_sort = get_coordinates_and_sorts(vs)
            M_x, D2_x, _ = FD_matrices(x_dofs[0][idx_sort[0]])
            M_y, D2_y, _ = FD_matrices(x_dofs[1][idx_sort[1]])
            M_q, _, _ = FD_matrices(x_dofs[2][idx_sort[2]])
            M_u, _, _ = FD_matrices(x_dofs[3][idx_sort[3]])
            # save resorted matrices
            param['M_x'], param['D2_x'] = M_x[idx_sort[0], :][:, idx_sort[0]], D2_x[idx_sort[0], :][:,
                                                                                         idx_sort[0]]
            param['M_y'], param['D2_y'] = M_y[idx_sort[1], :][:, idx_sort[1]], D2_y[idx_sort[1], :][:,
                                                                                         idx_sort[1]]
            param['M_q'], param['M_u'] = M_q[idx_sort[2], :][:, idx_sort[2]], M_u[idx_sort[3], :][:,
                                                                                        idx_sort[3]]
            param['bc_idx'] = np.array(
                [np.where(x_dofs[0] == 0)[0], (np.where(x_dofs[0] == param['lx'])[0])]).flatten()

            ass_rhs = problem_assemble_rhs_FD
            ass_lhs = problem_assemble_lhs_FD
            solve_modes = ["FD", "FD", "FD", "FD"]

        else:
            ass_rhs = None
            ass_lhs = None
            solve_modes = None
            print('not a valid type')

        pgd_prob = PGDProblem1(name='test_x_y_q_u00', name_coord=['X', 'Y', 'q', 'u0'],
                               modes_info=['T', 'Node', 'Scalar'],
                               Vs=vs, dom=0, bc_fct=create_bc, load=[qx, qy, qq, qu0],
                               param=param, rhs_fct=ass_rhs,
                               lhs_fct=ass_lhs, probs=['r', 's', 't', 'u'],
                               seq_fp=np.arange(len(vs)), PGD_nmax=1)
        if _type == 'FD':
            pgd_prob.MM = [param['M_x'], param['M_y'], param['M_q'], param['M_u']] # for norms!
        pgd_prob.stop_fp = 'chady'
        pgd_prob.max_fp_it = 50
        pgd_prob.tol_fp_it = 1e-5  # 1e-5
        # pgd_prob.fp_init = 'randomized' #?

        pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes)

        print(pgd_prob.simulation_info)
        print('PGD Amplitude', pgd_prob.amplitude)
        print('norms', pgd_prob.alpha)

        pgd_s = pgd_prob.return_PGD()  # as PGD class instance

        return pgd_s, param

class FEM_reference():

    def __init__(self, param=[], elem=[], ord=[]):

        self.param = param  # Parameters
        self.elem = elem

        # generate 2D mesh and function space
        self.mesh = df.RectangleMesh(df.Point(0,0),df.Point(self.param["lx"],self.param["ly"]), self.elem[0], self.elem[1])
        self.V = df.FunctionSpace(self.mesh,'CG',ord[0]+1) # second order breaked down 1.order in pgd split

        self.u00 = df.Constant(1.)
        self.q00 = df.Constant(1.)
        #
        self.bcExpr = df.Expression('u00*(1. - 1./3.*x[0])', degree=1, u00=self.u00)
        self.Q = df.Expression('x[0]<L/2 ? q00 : 0',degree=1,L=self.param["lx"],q00=self.q00)
        #
        v=df.TestFunction(self.V)
        T=df.TrialFunction(self.V)
        self.a = self.param["k"]*df.inner(df.grad(v),df.grad(T))*df.dx() # SIGN????
        self.l = v*self.Q*df.dx()
        #

    def __call__(self,values):
        # solve problem for requested values
        self.u00.assign(values[2])
        self.q00.assign(values[1])

        #boundary condition
        def leftright(x, on_boundary):
            return on_boundary and df.near(x[0], 0.0, 1e-6) or df.near(x[0], self.param["lx"], 1e-6)

        bc = df.DirichletBC(self.V, self.bcExpr, leftright)

        T = df.Function(self.V)
        df.solve(self.a==self.l, T, bcs=bc)

        # full 2D solution field
        #print(T.vector()[:])

        # solution at given y value
        # bla = self.V.tabulate_dof_coordinates()[:]
        # use regular x values for extracting solution
        x_x = np.linspace(0,self.param['lx'],self.elem[0]+1)
        T_x = np.zeros(len(x_x))
        for i in range(len(x_x)):
            T_x[i] = T((x_x[i], values[0]))

        return T_x, x_x, T



class problem(unittest.TestCase):

    def setUp(self):

        # problem parameters
        self.param = {"k": 0.5, "lx": 3, "ly": 3} # global parameters

        self.ranges = [[0., 3.], #xmin, xmax
                       [0., 3.], #ymin, ymax
                       [0., 50.], #qmin, qmax
                       [10., 50.]] #u0min, u0max
        self.ord = [1, 1, 1, 1] # order of each variable (for FEM use the first one)
        self.elem = [30,20,100,40] # number of elements for each variable
        # self.elem = [5, 5, 10, 10]  # number of elements for each variable

        self.fixed_dim = 0
        self.values = [1.5, 50, 10] # test evaluation[y,q,u0]

        self.plotting = True
        # self.plotting = False


    def TearDown(self):
        pass

    def test_solver(self):

        meshes, vs = create_meshes(self.elem, self.ord, self.ranges)

        # PGD FEM
        pgd_fem, param = create_PGD(param=self.param,vs=vs,_type="FEM")

        # PGD FD
        pgd_fd, param = create_PGD(param=self.param,vs=vs,_type="FD")

        input()

        if self.plotting:
            #### plotting optional
            import matplotlib.pyplot as plt

            # FEM reference solution 2D Problem at given values self.values
            ufem_x, fem_x, ufem_xy = FEM_reference(param=self.param, elem=self.elem, ord=self.ord)(self.values)

            # 2D plot just for FEM
            # df.plot(ufem_xy)
            # plt.show()

            upgd_fem = pgd_fem.evaluate(self.fixed_dim, [1, 2, 3], self.values, 0)
            upgd_fem_bc = upgd_fem.compute_vertex_values()[:] + \
                          param['BC_x'].compute_vertex_values()[:] * param['BC_y'](self.values[0]) * param["BC_q"](
                self.values[1]) * param["BC_u0"](self.values[2])

            upgd_fd = pgd_fd.evaluate(self.fixed_dim, [1, 2, 3], self.values, 0)
            upgd_fd_bc = upgd_fd.compute_vertex_values()[:] + \
                         param['BC_x'].compute_vertex_values()[:] * param['BC_y'](self.values[0]) * param["BC_q"](
                self.values[1]) * param["BC_u0"](self.values[2])

            plt.figure(1)
            plt.plot(fem_x,ufem_x,'-ob',label=f"FEM [y,q,u0]={self.values}")
            plt.plot(upgd_fem.function_space().mesh().coordinates()[:], upgd_fem_bc, '-*r',
                     label=f"PGD FEM [y,q,u0]={self.values}")
            plt.plot(upgd_fd.function_space().mesh().coordinates()[:], upgd_fd_bc, '-+y',
                     label=f"PGD FD [y,q,u0]={self.values}")
            plt.xlabel('x')
            plt.ylabel('T')
            plt.legend()
            plt.show()

        # checking PGD FD and PGD FEM same and to FEM?
        # check PGD to FEM nor with PGDErrorComputation because boundary condition not in PGD evaluate fct!!
        errors_PGD, errors_FEM1, errors_FEM2 = list(), list(), list()
        for _ in range(10):
            check_values = [self.ranges[1][0]+np.random.random()*(self.ranges[1][1]-self.ranges[1][0]),
                            self.ranges[2][0]+np.random.random()*(self.ranges[2][1]-self.ranges[2][0]),
                            self.ranges[3][0]+np.random.random()*(self.ranges[3][1]-self.ranges[3][0])]
            u1 = pgd_fd.evaluate(self.fixed_dim,[1,2,3],check_values,0).compute_vertex_values()[:]+\
                      param['BC_x'].compute_vertex_values()[:]*param['BC_y'](check_values[0])*param["BC_q"](check_values[1])*param["BC_u0"](check_values[2])
            u2 = pgd_fem.evaluate(self.fixed_dim,[1,2,3],check_values,0).compute_vertex_values()[:]+\
                      param['BC_x'].compute_vertex_values()[:]*param['BC_y'](check_values[0])*param["BC_q"](check_values[1])*param["BC_u0"](check_values[2])
            u3, _, _ = FEM_reference(param=self.param,elem=self.elem,ord=self.ord)(check_values)

            errors_PGD.append(np.linalg.norm(u2-u1)/np.linalg.norm(u1)) #PGD FEM - PGD FD
            errors_FEM1.append(np.linalg.norm(u1-u3)/np.linalg.norm(u3)) #PGD FD - FEM
            errors_FEM2.append(np.linalg.norm(u2-u3)/np.linalg.norm(u3)) #PGD FEM - FEM

        print(errors_PGD, np.mean(errors_PGD))
        print(errors_FEM1, np.mean(errors_FEM1))
        print(errors_FEM1, np.mean(errors_FEM2))
        self.assertTrue(np.mean(errors_PGD) < 5e-4)
        self.assertTrue(np.mean(errors_FEM1) < 5e-4)
        self.assertTrue(np.mean(errors_FEM2) < 1e-5)



if __name__ == '__main__':
    df.set_log_level(df.LogLevel.ERROR)

    import logging
    logging.basicConfig(level=logging.DEBUG)

    unittest.main()