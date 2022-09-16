'''
    3D PGD example (heat equation with a moving goldak heat source) with six PGD variables (space (x,y,z), time, heat input and heat transfer coefficient)
    solving PGD problem using FD non-space and FEM space
    returning PGDModel (as forward model) or PGD instance
'''

import unittest
import dolfin
import numpy as np

from pgdrome.solver import PGDProblem1, FD_matrices


def create_meshes_PGD(num_elem, ord, ranges):
    '''
    :param num_elem: list for each PG CO
    :param ord: list for each PG CO
    :param ranges: list for each PG CO
    :return: meshes and V
    '''

    print('create meshes PGD')

    meshes = list()
    Vs = list()

    dim = len(num_elem)

    for i in range(dim):
        mesh_tmp = dolfin.IntervalMesh(num_elem[i], ranges[i][0], ranges[i][1])
        Vs_tmp = dolfin.FunctionSpace(mesh_tmp, 'CG', ord[i])

        meshes.append(mesh_tmp)
        Vs.append(Vs_tmp)

    return meshes, Vs

def create_meshes_FEM(num_elem, ord, ranges):
    '''
    :param num_elem: list for each PG CO
    :param ord: list for each PG CO
    :param ranges: list for each PG CO
    :return: meshes and V
    '''

    print('create meshes FEM')

    meshes = list()
    Vs = list()

    # space mesh
    mesh_tmp = dolfin.BoxMesh(dolfin.Point(ranges[0][0], ranges[1][0], ranges[2][0]),\
                              dolfin.Point(ranges[0][1], ranges[1][1], ranges[2][1]),\
                              num_elem[0], num_elem[1], num_elem[2])
    Vs_tmp = dolfin.FunctionSpace(mesh_tmp,'CG', ord[0]*2)

    meshes.append(mesh_tmp)
    Vs.append(Vs_tmp)

    # time mesh
    mesh_tmp = dolfin.IntervalMesh(num_elem[3], ranges[3][0], ranges[3][1])
    Vs_tmp = dolfin.FunctionSpace(mesh_tmp, 'CG', ord[3])

    meshes.append(mesh_tmp)
    Vs.append(Vs_tmp)

    return meshes, Vs

def create_dom(Vs,param):
    # create domains in s
    subdomains = dolfin.MeshFunction("size_t", Vs[0].mesh(), Vs[0].mesh().topology().dim())  # same dimension
    subdomains.set_all(0)

    class left_dom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x < 1 + 1e-8

    class middle_dom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return 1 - 1e-8 <= x <= 2 + 1e-8

    class right_dom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x > 2 - 1e-8

    left_dom = left_dom()
    left_dom.mark(subdomains, 0)
    middle_dom = middle_dom()
    middle_dom.mark(subdomains, 1)
    right_dom = right_dom()
    right_dom.mark(subdomains, 2)

    # create boundarydomains
    boundarydomain_s = dolfin.MeshFunction("size_t", Vs[0].mesh(), Vs[1].mesh().topology().dim() - 1)
    boundarydomain_s.set_all(1)
    boundarydomain_y = dolfin.MeshFunction("size_t", Vs[1].mesh(), Vs[1].mesh().topology().dim() - 1)
    boundarydomain_y.set_all(0)
    boundarydomain_z = dolfin.MeshFunction("size_t", Vs[2].mesh(), Vs[2].mesh().topology().dim() - 1)
    boundarydomain_z.set_all(0)

    class left(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return np.isclose(x, np.min(Vs[0].mesh().coordinates()[:]))

    class right(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return np.isclose(x, np.max(Vs[0].mesh().coordinates()[:]))

    class front_back(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return np.isclose(x, np.min(Vs[1].mesh().coordinates()[:])) or np.isclose(x, np.max(
                Vs[1].mesh().coordinates()[:]))

    class top_bottom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return np.isclose(x, np.min(Vs[2].mesh().coordinates()[:])) or np.isclose(x, np.max(
                Vs[2].mesh().coordinates()[:]))

    left_ = left()
    left_.mark(boundarydomain_s, 0)
    right_ = right()
    right_.mark(boundarydomain_s, 2)
    front_back_dom = front_back()
    front_back_dom.mark(boundarydomain_y, 1)
    top_bottom_dom = top_bottom()
    top_bottom_dom.mark(boundarydomain_z, 1)

    dom = [subdomains, boundarydomain_y, boundarydomain_z, boundarydomain_s]
    dom.extend(np.zeros(len(Vs)-1, dtype=int))

    return dom

def create_bc(Vs,dom,param):
    # Initial condition
    def init(x, on_boundary):
        return x < np.min(Vs[3].mesh().coordinates()[:]) + 1E-8

    initCond = dolfin.DirichletBC(Vs[3], dolfin.Constant(0.), init)

    return [0, 0, 0, initCond, 0, 0]  # s, y, z, r, eta, h

def problem_assemble_lhs(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    # problem description left hand side of DGL for each fixed point problem

    # define measures
    ds_s = dolfin.Measure('ds', domain=meshes[0], subdomain_data=dom[3])
    ds_y = dolfin.Measure('ds', domain=meshes[1], subdomain_data=dom[1])
    ds_z = dolfin.Measure('ds', domain=meshes[2], subdomain_data=dom[2])
    dx_s = dolfin.Measure('dx', domain=meshes[0], subdomain_data=dom[0])

    # define expressions
    Bt = param["Bt"]
    Bx = param["Bx"]
    det_J = param["det_J"]

    alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ Fs[4].vector()[:]
    alpha_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ Fs[5].vector()[:]
    alpha_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ Fs[5].vector()[:]

    a = 0
    if typ == 's':
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ Fs[3].vector()[:]
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]

            a +=  dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r1
                * alpha_eta1
                * alpha_h1) \
                * param['a_s'] * param["rho"] * param["c_p"] * var_F * Bt[i][0][0] * fct_F.dx(0) * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r2
                * alpha_eta1
                * alpha_h1) \
                * param['a_r'] * param["rho"] * param["c_p"] * var_F * fct_F * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r3
                * alpha_eta1
                * alpha_h1) \
                * param['a_s2'] * param["k"] * var_F.dx(0) * fct_F.dx(0) * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_y2'] * param["k"] * var_F * fct_F * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_z2'] * param["k"] * var_F * fct_F * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_yz'] * var_F * fct_F * ds_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * ds_y)
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xz'] * var_F * fct_F * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * Fs[2] * ds_z)
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xy'] * var_F * fct_F * dx_s(i)

    if typ == 'y':
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ Fs[3].vector()[:]
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]

            a +=  dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0].dx(0) * Bt[i][0][0] * dx_s(i))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r1
                * alpha_eta1
                * alpha_h1) \
                * param['a_s'] * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r2
                * alpha_eta1
                * alpha_h1) \
                * param['a_r'] * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r3
                * alpha_eta1
                * alpha_h1) \
                * param['a_s2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_y2'] * param["k"] * var_F.dx(0) * fct_F.dx(0) * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_z2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * ds_s(i))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_yz'] * var_F * fct_F * dolfin.dx(meshes[1])  \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xz'] * var_F * fct_F * ds_y \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * Fs[2] * ds_z)
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xy'] * var_F * fct_F * dolfin.dx(meshes[1])

    if typ == 'z':
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ Fs[3].vector()[:]
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]

            a +=  dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0].dx(0) * Bt[i][0][0] * dx_s(i))
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * alpha_r1
                * alpha_eta1
                * alpha_h1) \
                * param['a_s'] * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * alpha_r2
                * alpha_eta1
                * alpha_h1) \
                * param['a_r'] * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i))
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * alpha_r3
                * alpha_eta1
                * alpha_h1) \
                * param['a_s2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_y2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_z2'] * param["k"] * var_F.dx(0) * fct_F.dx(0) * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * ds_s(i))
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_yz'] * var_F * fct_F * dolfin.dx(meshes[2])  \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * Fs[1] * ds_y)
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xz'] * var_F * fct_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xy'] * var_F * fct_F * ds_z

    if typ == 'r':
        for i in range(3):
            a +=  dolfin.assemble(Fs[0] * Bt[i][0][0] * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_s'] * param["rho"] * param["c_p"] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:] * param['M_r'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_r'] * param["rho"] * param["c_p"] * Bt[i][1] * param['D1_up_r'] \
                + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_s2'] * param["k"] * Bx[i].vector()[:] * Bx[i].vector()[:] * det_J[i].vector()[:] * param['M_r'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_y2'] * param["k"] * det_J[i].vector()[:] * param['M_r'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_z2'] * param["k"] * det_J[i].vector()[:] * param['M_r'] \
                + dolfin.assemble(Fs[0] * Fs[0] * ds_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h2 \
                * param['a_conv_yz'] * det_J[i].vector()[:] * param['M_r'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * ds_y) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h2 \
                * param['a_conv_xz'] * det_J[i].vector()[:] * param['M_r'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * ds_z) \
                * alpha_eta1 \
                * alpha_h2 \
                * param['a_conv_xy'] * det_J[i].vector()[:] * param['M_r']

        # add initial condition
        a[:, param['bc_idx']] = 0
        a[param['bc_idx'], :] = 0
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
                * alpha_h1 \
                * param['a_s'] * param["rho"] * param["c_p"] * param['M_eta'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * alpha_h1 \
                * param['a_r'] * param["rho"] * param["c_p"] * param['M_eta'] \
                + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * alpha_h1 \
                * param['a_s2'] * param["k"] * param['M_eta'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_h1 \
                * param['a_y2'] * param["k"] * param['M_eta'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_h1 \
                * param['a_z2'] * param["k"] * param['M_eta'] \
                + dolfin.assemble(Fs[0] * Fs[0] * ds_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_h2 \
                * param['a_conv_yz'] * param['M_eta'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * ds_y) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_h2 \
                * param['a_conv_xz'] * param['M_eta'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * ds_z) \
                * alpha_r4 \
                * alpha_h2 \
                * param['a_conv_xy'] * param['M_eta']

    if typ == 'h':
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ Fs[3].vector()[:]
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ Fs[3].vector()[:]

            a +=  dolfin.assemble(Fs[0] * Bt[i][0][0] * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r1 \
                * alpha_eta1 \
                * param['a_s'] * param["rho"] * param["c_p"] * param['M_h'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * alpha_eta1 \
                * param['a_r'] * param["rho"] * param["c_p"] * param['M_h'] \
                + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * alpha_eta1 \
                * param['a_s2'] * param["k"] * param['M_h'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_y2'] * param["k"] * param['M_h'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_z2'] * param["k"] * param['M_h'] \
                + dolfin.assemble(Fs[0] * Fs[0] * ds_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_conv_yz'] * param['h'].vector()[:] * param['M_h'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * ds_y) \
                * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_conv_xz'] * param['h'].vector()[:] * param['M_h'] \
                + dolfin.assemble(Fs[0] * Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * Fs[2] * ds_z) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_conv_xy'] * param['h'].vector()[:] * param['M_h']

    return a

def problem_assemble_rhs(fct_F,var_F,Fs,meshes,dom,param,Q,PGD_func,typ,nE,dim):
    # problem description right hand side of DGL for each fixed point problem

    # define measures
    ds_s = dolfin.Measure('ds', domain=meshes[0], subdomain_data=dom[3])
    ds_y = dolfin.Measure('ds', domain=meshes[1], subdomain_data=dom[1])
    ds_z = dolfin.Measure('ds', domain=meshes[2], subdomain_data=dom[2])
    dx_s = dolfin.Measure('dx', domain=meshes[0], subdomain_data=dom[0])

    # define expressions
    Bt = param["Bt"]
    Bx = param["Bx"]
    det_J = param["det_J"]
    IC = [param["IC_s"], param["IC_y"], param["IC_z"], param["IC_r"], param["IC_eta"], param["IC_h"]]

    beta_r1 = (Fs[3].vector()[:] * det_J[1].vector()[:]).transpose() @ param['M_r'] @ Q[3].vector()[:]
    beta_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ Q[4].vector()[:]
    beta_eta2 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ np.ones(len(Q[4].vector()))
    beta_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ Q[5].vector()[:]
    beta_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ np.ones(len(Q[5].vector()))
    alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ IC[4].vector()[:]
    alpha_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ IC[5].vector()[:]
    alpha_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ IC[5].vector()[:]

    bc = 0
    if typ == 's':
        for i in range(3):
            beta_r2 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ np.ones(len(Q[3].vector()))
            bc += dolfin.Constant(dolfin.assemble(Fs[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2]))
                * beta_r2
                * beta_eta2
                * beta_h2) \
                * param['l2_yz'] * param["T_amb"] * var_F * ds_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * ds_y)
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2]))
                * beta_r2
                * beta_eta2
                * beta_h2) \
                * param['l2_xz'] * param["T_amb"] * var_F * dx_s(i) \
                + dolfin.Constant(dolfin.assemble(Fs[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * ds_z)
                * beta_r2
                * beta_eta2
                * beta_h2) \
                * param['l2_xy'] * param["T_amb"] * var_F * dx_s(i)
        l =   dolfin.Constant(dolfin.assemble(Fs[1] * Q[1] * dolfin.dx(meshes[1]))
            * dolfin.assemble(Fs[2] * Q[2] * dolfin.dx(meshes[2]))
            * beta_r1
            * beta_eta1
            * beta_h1) \
            * param['l1'] * var_F * Q[0] * dx_s(1) \
            + bc
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ IC[3].vector()[:]
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]

            l +=- dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r1
                * alpha_eta1
                * alpha_h1) \
                * param['a_s'] * param["rho"] * param["c_p"] * var_F * Bt[i][0][0] * IC[0].dx(0) * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r2
                * alpha_eta1
                * alpha_h1) \
                * param['a_r'] * param["rho"] * param["c_p"] * var_F * IC[0] * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r3
                * alpha_eta1
                * alpha_h1) \
                * param['a_s2'] * param["k"] * var_F.dx(0) * IC[0].dx(0) * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_y2'] * param["k"] * var_F * IC[0] * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2].dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_z2'] * param["k"] * var_F * IC[0] * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_yz'] * var_F * IC[0] * ds_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * ds_y)
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xz'] * var_F * IC[0] * dx_s(i) \
                - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * dolfin.assemble(Fs[2] * IC[2] * ds_z)
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xy'] * var_F * IC[0] * dx_s(i)
        if nE > 0:
            for old in range(nE):
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                for j in range(3):
                    alpha_old_r1 = (Fs[3].vector()[:] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r2 = (Fs[3].vector()[:] * Bt[j][1]).transpose() @ param['D1_up_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r3 = (Fs[3].vector()[:] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r4 = (Fs[3].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]

                    l +=- dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r1
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_s'] * param["rho"] * param["c_p"] * var_F * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r2
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_r'] * param["rho"] * param["c_p"] * var_F * PGD_func[0][old] * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r3
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_s2'] * param["k"] * var_F.dx(0) * PGD_func[0][old].dx(0) * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1]))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_y2'] * param["k"] * var_F * PGD_func[0][old] * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_z2'] * param["k"] * var_F * PGD_func[0][old] * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h2) \
                        * param['a_conv_yz'] * var_F * PGD_func[0][old] * ds_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y)
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h2) \
                        * param['a_conv_xz'] * var_F * PGD_func[0][old] * dx_s(j) \
                        - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z)
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h2) \
                        * param['a_conv_xy'] * var_F * PGD_func[0][old] * dx_s(j)

    if typ == 'y':
        for i in range(3):
            beta_r2 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ np.ones(len(Q[3].vector()))
            bc += dolfin.Constant(dolfin.assemble(Fs[0] * ds_s(i))
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2]))
                * beta_r2
                * beta_eta2
                * beta_h2) \
                * param['l2_yz'] * param["T_amb"] * var_F * dolfin.dx(meshes[1]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2]))
                * beta_r2
                * beta_eta2
                * beta_h2) \
                * param['l2_xz'] * param["T_amb"] * var_F * ds_y \
                + dolfin.Constant(dolfin.assemble(Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * ds_z)
                * beta_r2
                * beta_eta2
                * beta_h2) \
                * param['l2_xy'] * param["T_amb"] * var_F * dolfin.dx(meshes[1])
        l =   dolfin.Constant(dolfin.assemble(Fs[0] * Q[0] * dx_s(1))
            * dolfin.assemble(Fs[2] * Q[2] * dolfin.dx(meshes[2]))
            * beta_r1
            * beta_eta1
            * beta_h1) \
            * param['l1'] * var_F * Q[1] * dolfin.dx(meshes[1]) \
            + bc
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ IC[3].vector()[:]
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]

            l +=- dolfin.Constant(dolfin.assemble(Fs[0] * Bt[i][0][0] * IC[0].dx(0) * dx_s(i))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r1
                * alpha_eta1
                * alpha_h1) \
                * param['a_s'] * param["rho"] * param["c_p"] * var_F * IC[1] * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r2
                * alpha_eta1
                * alpha_h1) \
                * param['a_r'] * param["rho"] * param["c_p"] * var_F * IC[1] * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * IC[0].dx(0) * dx_s(i))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r3
                * alpha_eta1
                * alpha_h1) \
                * param['a_s2'] * param["k"] * var_F * IC[1] * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_y2'] * param["k"] * var_F.dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[2].dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_z2'] * param["k"] * var_F * IC[1] * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * ds_s(i))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_yz'] * var_F * IC[1] * dolfin.dx(meshes[1]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xz'] * var_F * IC[1] * ds_y \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[2] * IC[2] * ds_z)
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xy'] * var_F * IC[1] * dolfin.dx(meshes[1])
        if nE > 0:
            for old in range(nE):
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                for j in range(3):
                    alpha_old_r1 = (Fs[3].vector()[:] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r2 = (Fs[3].vector()[:] * Bt[j][1]).transpose() @ param['D1_up_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r3 = (Fs[3].vector()[:] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r4 = (Fs[3].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]

                    l +=- dolfin.Constant(dolfin.assemble(Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r1
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_s'] * param["rho"] * param["c_p"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r2
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_r'] * param["rho"] * param["c_p"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r3
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_s2'] * param["k"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_y2'] * param["k"] * var_F.dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_z2'] * param["k"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_s(j))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h2) \
                        * param['a_conv_yz'] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h2) \
                        * param['a_conv_xz'] * var_F * PGD_func[1][old] * ds_y \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z)
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h2) \
                        * param['a_conv_xy'] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1])

    if typ == 'z':
        for i in range(3):
            beta_r2 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ np.ones(len(Q[3].vector()))
            bc += dolfin.Constant(dolfin.assemble(Fs[0] * ds_s(i))
                * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1]))
                * beta_r2
                * beta_eta2
                * beta_h2) \
                * param['l2_yz'] * param["T_amb"] * var_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * ds_y)
                * beta_r2
                * beta_eta2
                * beta_h2) \
                * param['l2_xz'] * param["T_amb"] * var_F * dolfin.dx(meshes[2]) \
                + dolfin.Constant(dolfin.assemble(Fs[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1]))
                * beta_r2
                * beta_eta2
                * beta_h2) \
                * param['l2_xy'] * param["T_amb"] * var_F * ds_z
        l =   dolfin.Constant(dolfin.assemble(Fs[0] * Q[0] * dx_s(1))
            * dolfin.assemble(Fs[1] * Q[1] * dolfin.dx(meshes[1]))
            * beta_r1
            * beta_eta1
            * beta_h1) \
            * param['l1'] * var_F * Q[2] * dolfin.dx(meshes[2]) \
            + bc
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ IC[3].vector()[:]
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]

            l +=- dolfin.Constant(dolfin.assemble(Fs[0] * Bt[i][0][0] * IC[0].dx(0) * dx_s(i))
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * alpha_r1
                * alpha_eta1
                * alpha_h1) \
                * param['a_s'] * param["rho"] * param["c_p"] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * alpha_r2
                * alpha_eta1
                * alpha_h1) \
                * param['a_r'] * param["rho"] * param["c_p"] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * IC[0].dx(0) * dx_s(i))
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * alpha_r3
                * alpha_eta1
                * alpha_h1) \
                * param['a_s2'] * param["k"] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[1].dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_y2'] * param["k"] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * alpha_r4
                * alpha_eta1
                * alpha_h1) \
                * param['a_z2'] * param["k"] * var_F.dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * ds_s(i))
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_yz'] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * IC[1] * ds_y)
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xz'] * var_F * IC[2] * dolfin.dx(meshes[2]) \
                - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0] * dx_s(i))
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1]))
                * alpha_r4
                * alpha_eta1
                * alpha_h2) \
                * param['a_conv_xy'] * var_F * IC[2] * ds_z
        if nE > 0:
            for old in range(nE):
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                for j in range(3):
                    alpha_old_r1 = (Fs[3].vector()[:] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r2 = (Fs[3].vector()[:] * Bt[j][1]).transpose() @ param['D1_up_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r3 = (Fs[3].vector()[:] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r4 = (Fs[3].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]

                    l +=- dolfin.Constant(dolfin.assemble(Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j))
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * alpha_old_r1
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_s'] * param["rho"] * param["c_p"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * alpha_old_r2
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_r'] * param["rho"] * param["c_p"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j))
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * alpha_old_r3
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_s2'] * param["k"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_y2'] * param["k"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h1) \
                        * param['a_z2'] * param["k"] * var_F.dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_s(j))
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h2) \
                        * param['a_conv_yz'] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y)
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h2) \
                        * param['a_conv_xz'] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                        - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j))
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                        * alpha_old_r4
                        * alpha_old_eta1
                        * alpha_old_h2) \
                        * param['a_conv_xy'] * var_F * PGD_func[2][old] * ds_z

    if typ == 'r':
        for i in range(3):
            bc += dolfin.assemble(Fs[0] * ds_s(i)) \
                * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
                * beta_eta2 \
                * beta_h2 \
                * param['l2_yz'] * param["T_amb"] * det_J[i].vector()[:] * param['M_r'] @ np.ones(len(Q[3].vector())) \
                + dolfin.assemble(Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * ds_y) \
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
                * beta_eta2 \
                * beta_h2 \
                * param['l2_xz'] * param["T_amb"] * det_J[i].vector()[:] * param['M_r'] @ np.ones(len(Q[3].vector())) \
                + dolfin.assemble(Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * ds_z) \
                * beta_eta2 \
                * beta_h2 \
                * param['l2_xy'] * param["T_amb"] * det_J[i].vector()[:] * param['M_r'] @ np.ones(len(Q[3].vector()))
        l =   dolfin.assemble(Fs[0] * Q[0] * dx_s(1)) \
            * dolfin.assemble(Fs[1] * Q[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Q[2] * dolfin.dx(meshes[2])) \
            * beta_eta1 \
            * beta_h1 \
            * param['l1'] * det_J[1].vector()[:] * param['M_r'] @ Q[3].vector()[:] \
            + bc
        for i in range(3):
            l +=- dolfin.assemble(Fs[0] * Bt[i][0][0] * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_s'] * param["rho"] * param["c_p"] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:] * param['M_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_r'] * param["rho"] * param["c_p"] * Bt[i][1] * param['D1_up_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0].dx(0) * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_s2'] * param["k"] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:] * param['M_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_y2'] * param["k"] * det_J[i].vector()[:] * param['M_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h1 \
                * param['a_z2'] * param["k"] * det_J[i].vector()[:] * param['M_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * ds_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h2 \
                * param['a_conv_yz'] * det_J[i].vector()[:] * param['M_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * ds_y) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_eta1 \
                * alpha_h2 \
                * param['a_conv_xz'] * det_J[i].vector()[:] * param['M_r'] @ IC[3].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * ds_z) \
                * alpha_eta1 \
                * alpha_h2 \
                * param['a_conv_xy'] * det_J[i].vector()[:] * param['M_r'] @ IC[3].vector()[:]
        if nE > 0:
            for old in range(nE):
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                for j in range(3):
                    l +=- dolfin.assemble(Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_eta1 \
                        * alpha_old_h1 \
                        * param['a_s'] * param["rho"] * param["c_p"] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_eta1 \
                        * alpha_old_h1 \
                        * param['a_r'] * param["rho"] * param["c_p"] * Bt[j][1] * param['D1_up_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_eta1 \
                        * alpha_old_h1 \
                        * param['a_s2'] * param["k"] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_eta1 \
                        * alpha_old_h1 \
                        * param['a_y2'] * param["k"] * det_J[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                        * alpha_old_eta1 \
                        * alpha_old_h1 \
                        * param['a_z2'] * param["k"] * det_J[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_eta1 \
                        * alpha_old_h2 \
                        * param['a_conv_yz'] * det_J[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_eta1 \
                        * alpha_old_h2 \
                        * param['a_conv_xz'] * det_J[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z) \
                        * alpha_old_eta1 \
                        * alpha_old_h2 \
                        * param['a_conv_xy'] * det_J[j].vector()[:] * param['M_r'] @ PGD_func[3][old].vector()[:]

        # add initial condition
        l[param['bc_idx']] = 0

    if typ == 'eta':
        for i in range(3):
            beta_r2 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ np.ones(len(Q[3].vector()))
            bc += dolfin.assemble(Fs[0] * ds_s(i)) \
                * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
                * beta_r2 \
                * beta_h2 \
                * param['l2_yz'] * param["T_amb"] * param['M_eta'] @ np.ones(len(Q[4].vector())) \
                + dolfin.assemble(Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * ds_y) \
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
                * beta_r2 \
                * beta_h2 \
                * param['l2_xz'] * param["T_amb"] * param['M_eta'] @ np.ones(len(Q[4].vector())) \
                + dolfin.assemble(Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * ds_z) \
                * beta_r2 \
                * beta_h2 \
                * param['l2_xy'] * param["T_amb"] * param['M_eta'] @ np.ones(len(Q[4].vector()))
        l =   dolfin.assemble(Fs[0] * Q[0] * dx_s(1)) \
            * dolfin.assemble(Fs[1] * Q[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Q[2] * dolfin.dx(meshes[2])) \
            * beta_r1 \
            * beta_h1 \
            * param['l1'] * param['M_eta'] @ Q[4].vector()[:] \
            + bc
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ IC[3].vector()[:]
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]

            l +=- dolfin.assemble(Fs[0] * Bt[i][0][0] * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r1 \
                * alpha_h1 \
                * param['a_s'] * param["rho"] * param["c_p"] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * alpha_h1 \
                * param['a_r'] * param["rho"] * param["c_p"] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0].dx(0) * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * alpha_h1 \
                * param['a_s2'] * param["k"] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_h1 \
                * param['a_y2'] * param["k"] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_h1 \
                * param['a_z2'] * param["k"] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * ds_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_h2 \
                * param['a_conv_yz'] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * ds_y) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_h2 \
                * param['a_conv_xz'] * param['M_eta'] @ IC[4].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * ds_z) \
                * alpha_r4 \
                * alpha_h2 \
                * param['a_conv_xy'] * param['M_eta'] @ IC[4].vector()[:]
        if nE > 0:
            for old in range(nE):
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                for j in range(3):
                    alpha_old_r1 = (Fs[3].vector()[:] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r2 = (Fs[3].vector()[:] * Bt[j][1]).transpose() @ param['D1_up_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r3 = (Fs[3].vector()[:] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r4 = (Fs[3].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]

                    l +=- dolfin.assemble(Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r1 \
                        * alpha_old_h1 \
                        * param['a_s'] * param["rho"] * param["c_p"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r2 \
                        * alpha_old_h1 \
                        * param['a_r'] * param["rho"] * param["c_p"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r3 \
                        * alpha_old_h1 \
                        * param['a_s2'] * param["k"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_old_h1 \
                        * param['a_y2'] * param["k"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_old_h1 \
                        * param['a_z2'] * param["k"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_old_h2 \
                        * param['a_conv_yz'] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_old_h2 \
                        * param['a_conv_xz'] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z) \
                        * alpha_old_r4 \
                        * alpha_old_h2 \
                        * param['a_conv_xy'] * param['M_eta'] @ PGD_func[4][old].vector()[:]

    if typ == 'h':
        for i in range(3):
            beta_r2 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ np.ones(len(Q[3].vector()))
            bc += dolfin.assemble(Fs[0] * ds_s(i)) \
                * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
                * beta_r2 \
                * beta_eta2 \
                * param['l2_yz'] * param["T_amb"] * param['h'].vector()[:] * param['M_h'] @ np.ones(len(Q[5].vector())) \
                + dolfin.assemble(Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * ds_y) \
                * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
                * beta_r2 \
                * beta_eta2 \
                * param['l2_xz'] * param["T_amb"] * param['h'].vector()[:] * param['M_h'] @ np.ones(len(Q[5].vector())) \
                + dolfin.assemble(Fs[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * ds_z) \
                * beta_r2 \
                * beta_eta2 \
                * param['l2_xy'] * param["T_amb"] * param['h'].vector()[:] * param['M_h'] @ np.ones(len(Q[5].vector()))
        l =   dolfin.assemble(Fs[0] * Q[0] * dx_s(1)) \
            * dolfin.assemble(Fs[1] * Q[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Q[2] * dolfin.dx(meshes[2])) \
            * beta_r1 \
            * beta_eta1 \
            * param['l1'] * param['M_h'] @ Q[5].vector()[:] \
            + bc
        for i in range(3):
            alpha_r1 = (Fs[3].vector()[:] * Bt[i][0][1].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r2 = (Fs[3].vector()[:] * Bt[i][1]).transpose() @ param['D1_up_r'] @ IC[3].vector()[:]
            alpha_r3 = (Fs[3].vector()[:] * det_J[i].vector()[:] * Bx[i].vector()[:] * Bx[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]
            alpha_r4 = (Fs[3].vector()[:] * det_J[i].vector()[:]).transpose() @ param['M_r'] @ IC[3].vector()[:]

            l +=- dolfin.assemble(Fs[0] * Bt[i][0][0] * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r1 \
                * alpha_eta1 \
                * param['a_s'] * param["rho"] * param["c_p"] * param['M_h'] @ IC[5].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r2 \
                * alpha_eta1 \
                * param['a_r'] * param["rho"] * param["c_p"] * param['M_h'] @ IC[5].vector()[:] \
                - dolfin.assemble(Fs[0].dx(0) * IC[0].dx(0) * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r3 \
                * alpha_eta1 \
                * param['a_s2'] * param["k"] * param['M_h'] @ IC[5].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1].dx(0) * IC[1].dx(0) * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_y2'] * param["k"] * param['M_h'] @ IC[5].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2].dx(0) * IC[2].dx(0) * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_z2'] * param["k"] * param['M_h'] @ IC[5].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * ds_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_conv_yz'] * param['h'].vector()[:] * param['M_h'] @ IC[5].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * ds_y) \
                * dolfin.assemble(Fs[2] * IC[2] * dolfin.dx(meshes[2])) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_conv_xz'] * param['h'].vector()[:] * param['M_h'] @ IC[5].vector()[:] \
                - dolfin.assemble(Fs[0] * IC[0] * dx_s(i)) \
                * dolfin.assemble(Fs[1] * IC[1] * dolfin.dx(meshes[1])) \
                * dolfin.assemble(Fs[2] * IC[2] * ds_z) \
                * alpha_r4 \
                * alpha_eta1 \
                * param['a_conv_xy'] * param['h'].vector()[:] * param['M_h'] @ IC[5].vector()[:]
        if nE > 0:
            for old in range(nE):
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                for j in range(3):
                    alpha_old_r1 = (Fs[3].vector()[:] * Bt[j][0][1].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r2 = (Fs[3].vector()[:] * Bt[j][1]).transpose() @ param['D1_up_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r3 = (Fs[3].vector()[:] * det_J[j].vector()[:] * Bx[j].vector()[:] * Bx[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]
                    alpha_old_r4 = (Fs[3].vector()[:] * det_J[j].vector()[:]).transpose() @ param['M_r'] @ PGD_func[3][old].vector()[:]

                    l +=- dolfin.assemble(Fs[0] * Bt[j][0][0] * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r1 \
                        * alpha_old_eta1 \
                        * param['a_s'] * param["rho"] * param["c_p"] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r2 \
                        * alpha_old_eta1 \
                        * param['a_r'] * param["rho"] * param["c_p"] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                        - dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r3 \
                        * alpha_old_eta1 \
                        * param['a_s2'] * param["k"] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_old_eta1 \
                        * param['a_y2'] * param["k"] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_old_eta1 \
                        * param['a_z2'] * param["k"] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_old_eta1 \
                        * param['a_conv_yz'] * param['h'].vector()[:] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                        * alpha_old_r4 \
                        * alpha_old_eta1 \
                        * param['a_conv_xz'] * param['h'].vector()[:] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                        - dolfin.assemble(Fs[0] * PGD_func[0][old] * dx_s(j)) \
                        * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                        * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z) \
                        * alpha_old_r4 \
                        * alpha_old_eta1 \
                        * param['a_conv_xy'] * param['h'].vector()[:] * param['M_h'] @ PGD_func[5][old].vector()[:]

    return l

def create_PGD(param={}, vs=[], q_PGD=None, q_coeff=None):

    # define position functions
    param.update({"h_1": dolfin.interpolate(dolfin.Expression('x[0]*rref-(h_g/2)', degree=4, h_g=param["h_g"], rref=param['r_ref']),vs[3])})
    param.update({"h_2": dolfin.interpolate(dolfin.Expression('L-(h_g/2)-x[0]*rref', degree=4, h_g=param["h_g"], rref=param['r_ref'], L=param["L"]),vs[3])})

    # define derivates with respect to s/r and jacobian
    Bt = [[[dolfin.Expression('-vel*x[0]', degree=2, vel=param["vel"]), dolfin.interpolate(dolfin.Expression('1/h_1', degree=2, h_1=param["h_1"]), vs[3])],
               param["vel"]],
          [[dolfin.Constant(-param["vel"]), dolfin.interpolate(dolfin.Constant(1/param["h_g"]), vs[3])],
               param["vel"]],
          [[dolfin.Expression('vel*(x[0]-2)-vel', degree=2, vel=param["vel"]), dolfin.interpolate(dolfin.Expression('1/h_2', degree=2, h_2=param["h_2"]), vs[3])],
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

    # define inhomogeneous dirichlet IC
    param.update({'IC_s': dolfin.interpolate(param['IC_s'], vs[0])})
    param.update({'IC_y': dolfin.interpolate(param['IC_y'], vs[1])})
    param.update({'IC_z': dolfin.interpolate(param['IC_z'], vs[2])})
    param.update({'IC_r': dolfin.interpolate(param['IC_r'], vs[3])})
    param.update({'IC_eta': dolfin.interpolate(param['IC_eta'], vs[4])})
    param.update({'IC_h': dolfin.interpolate(param['IC_h'], vs[5])})

    # define inhomogeneous convection BC
    param.update({'h': dolfin.interpolate(dolfin.Expression('x[0] * href', href=param['h_ref'], degree=4), vs[5])})

    # define heat source in s, y, z, r, eta and h
    q_s = dolfin.interpolate(q_PGD[0], vs[0])
    q_y = dolfin.interpolate(q_PGD[1], vs[1])
    q_z = dolfin.interpolate(q_PGD[2], vs[2])
    q_r = dolfin.interpolate(dolfin.Expression('q_coeff', q_coeff=q_coeff, degree=1), vs[3])
    q_eta = dolfin.interpolate(dolfin.Expression('x[0]*P', P=param['P'], degree=1), vs[4])
    q_h = dolfin.interpolate(dolfin.Constant(1.), vs[5])

    # create FD matrices from meshes
    # r case
    r_dofs = np.array(vs[3].tabulate_dof_coordinates()[:].flatten()) 
    r_sort = np.argsort(r_dofs)
    M_r, _, D1_up_r = FD_matrices(r_dofs[r_sort])
    param['M_r'], param['D1_up_r'] = M_r[r_sort, :][:, r_sort], D1_up_r[r_sort, :][:, r_sort]
    param['bc_idx'] = np.where(r_dofs == 0)[0]
    # eta case
    eta_dofs = np.array(vs[4].tabulate_dof_coordinates()[:].flatten())
    eta_sort = np.argsort(eta_dofs)
    M_eta, _, _ = FD_matrices(eta_dofs[eta_sort])
    param['M_eta'] = M_eta[eta_sort, :][:, eta_sort]
    # h case
    h_dofs = np.array(vs[5].tabulate_dof_coordinates()[:].flatten())
    h_sort = np.argsort(h_dofs)
    M_h, _, _ = FD_matrices(h_dofs[h_sort])
    param['M_h'] = M_h[h_sort, :][:, h_sort]

    solve_modes = ["FEM", "FEM", "FEM", "FD", "FD", "FD"]

    pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-SYZREtaH', name_coord=['S', 'Y', 'Z', 'R', 'Eta', 'H'],
                           modes_info=['T', 'Node', 'Scalar'],
                           Vs=vs, dom_fct=create_dom, bc_fct=create_bc, load=[q_s, q_y, q_z, q_r, q_eta, q_h],
                           param=param, rhs_fct=problem_assemble_rhs, lhs_fct=problem_assemble_lhs,
                           probs=['s', 'y', 'z', 'r', 'eta', 'h'], seq_fp=np.arange(len(vs)),
                           PGD_nmax=20, PGD_tol=1e-5)

    pgd_prob.MM = [0, 0, 0, param['M_r'], param['M_eta'], param['M_h']]  # for norms!

    pgd_prob.stop_fp = 'norm'
    pgd_prob.max_fp_it = 50
    pgd_prob.tol_fp_it = 1e-5
    pgd_prob.norm_modes = 'stiff'

    pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes)
    # pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes, settings = {"preconditioner": "amg", "linear_solver": "gmres"})

    print(pgd_prob.simulation_info)
    print('PGD Amplitude', pgd_prob.amplitude)

    pgd_s = pgd_prob.return_PGD()  # as PGD class instance

    return pgd_s, param

def remapping(pgd_solution=None, param=None, pos_fixed=None, eta_fixed=None, h_fixed=None):
    # compute remapping of heating phase

    # initialize data
    r_mesh = pgd_solution.mesh[3].dataX
    time = r_mesh/param["vel"]

    x_fixed = pos_fixed[0]
    y_fixed = pos_fixed[1]
    z_fixed = pos_fixed[2]

    # map back to x in PGD
    PGD_heating = np.zeros(len(r_mesh))
    for i, rr in enumerate(r_mesh):
        u_pgd = pgd_solution.evaluate(0, [1, 2, 3, 4, 5], [y_fixed, z_fixed, rr, eta_fixed, h_fixed], 0)

        # left side
        s_temp = x_fixed * param['x_ref'] / param["h_1"](rr)
        if s_temp < 1:
            PGD_heating[i] = u_pgd(s_temp)
        else:
            # center
            s_temp = (x_fixed * param['x_ref'] - param["h_1"](rr)) / param["h_g"] + 1
            if s_temp <= 2:
                PGD_heating[i] = u_pgd(s_temp)
            else:
                # right side
                s_temp = (x_fixed * param['x_ref'] - param["h_1"](rr) - param["h_g"]) / param["h_2"](rr) + 2
                PGD_heating[i] = u_pgd(s_temp)

    # add initial condition to heating phase
    PGD_heating += param['IC_x'](x_fixed) * param['IC_y'](y_fixed) * param['IC_z'](z_fixed) \
        * param['IC_r'].compute_vertex_values()[:] * param["IC_eta"](eta_fixed) * param["IC_h"](h_fixed)

    return PGD_heating, time

# TEST: PGD result VS FEM result
#==============================================================================

# Finite Element Model
#=======================================
# define boundary's
class Bottom(dolfin.SubDomain):
    def __init__(self, param={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = param

    def inside(self, x, on_boundary):
        return on_boundary and dolfin.near(x[2], 0, 1E-8)

class Top(dolfin.SubDomain):
    def __init__(self, param={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = param

    def inside(self, x, on_boundary):
        return on_boundary and dolfin.near(x[2], self.param['H'], 1E-8)

class Left(dolfin.SubDomain):
    def __init__(self, param={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = param

    def inside(self, x, on_boundary):
        return on_boundary and dolfin.near(x[0], 0, 1E-8)

class Right(dolfin.SubDomain):
    def __init__(self, param={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = param

    def inside(self, x, on_boundary):
        return on_boundary and dolfin.near(x[0], self.param['L'], 1E-8)

class Front(dolfin.SubDomain):
    def __init__(self, param={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = param

    def inside(self, x, on_boundary):
        return on_boundary and dolfin.near(x[1], 0, 1E-8)

class Back(dolfin.SubDomain):
    def __init__(self, param={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = param

    def inside(self, x, on_boundary):
        return on_boundary and dolfin.near(x[1], self.param['W'], 1E-8)

# reference solution class
class Reference():

    def __init__(self, param={}, vs=[], q=None, pos_fixed=None):

        self.vs = vs # Location
        self.param = param # Parameters
        self.q = q # source term

        # time points
        self.time_mesh = self.vs[1].mesh().coordinates()[:]
        self.T_n = dolfin.interpolate(self.param["Tamb_fct"], self.vs[0])
        self.T_n.vector()[:] *= self.param['T_ref']
        self.Tamb = dolfin.Function(self.vs[0])
        self.Tamb.vector()[:] = 1 * self.T_n.vector()[:]

        # problem
        self.mesh = self.vs[0].mesh()
        T = dolfin.TrialFunction(self.vs[0])
        v = dolfin.TestFunction(self.vs[0])
        self.dt = dolfin.Constant(1.)
        self.Q = dolfin.Constant(1.)
        self.h = dolfin.Constant(1.)
        self.pos_fixed = (pos_fixed[0] * self.param['x_ref'], pos_fixed[1] * self.param['y_ref'], pos_fixed[2] * self.param['z_ref'])

        # Assign boundary conditions
        boundary_parts = dolfin.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        boundary_parts.set_all(0)  # marks whole domain as 0

        bottom = Bottom(self.param)
        bottom.mark(boundary_parts, 1)  # marks bottom of the domain as 1
        top = Top(self.param)
        top.mark(boundary_parts, 2)  # marks top of the domain as 2
        left = Left(self.param)
        left.mark(boundary_parts, 3)  # marks left side of the domain as 3
        right = Right(self.param)
        right.mark(boundary_parts, 4)  # marks right side of the domain as 4
        front = Front(self.param)
        front.mark(boundary_parts, 5)  # marks front of the domain as 5
        back = Back(self.param)
        back.mark(boundary_parts, 6)  # marks back of the domain as 6

        self.ds = dolfin.Measure('ds', domain=self.mesh, subdomain_data=boundary_parts)

        # collect variational form
        self.F = self.param["rho"] * self.param["c_p"] * T * v * dolfin.dx() \
                 + self.dt * self.param["k"] * dolfin.dot(dolfin.grad(T), dolfin.grad(v)) * dolfin.dx() \
                 - (self.dt * self.Q * self.q + self.param["rho"] * self.param["c_p"] * self.T_n) * v * dolfin.dx() \
                 + self.dt * self.h * (self.T_n - self.Tamb) * v * self.ds(1) \
                 + self.dt * self.h * (self.T_n - self.Tamb) * v * self.ds(2) \
                 + self.dt * self.h * (self.T_n - self.Tamb) * v * self.ds(3) \
                 + self.dt * self.h * (self.T_n - self.Tamb) * v * self.ds(4) \
                 + self.dt * self.h * (self.T_n - self.Tamb) * v * self.ds(5) \
                 + self.dt * self.h * (self.T_n - self.Tamb) * v * self.ds(6)

        self.a, self.L = dolfin.lhs(self.F), dolfin.rhs(self.F)

    def __call__(self, values):

        # check time mesh for requested time value
        if np.where(self.time_mesh == values[0] * self.param['t_ref'])[0].size == 0:
            print("ERROR time step not in mesh What to do?")
        self.Q.assign(values[1]*self.param["P"])
        self.h.assign(values[2])

        # Time-stepping
        Ttime = []
        Ttmp = dolfin.Function(self.vs[0])
        Ttmp.vector()[:] = 1 * self.T_n.vector()[:]
        Ttime.append(Ttmp)  # otherwise it will be overwritten with new solution
        Txfixed = [np.copy(self.T_n(self.pos_fixed))]
        T = dolfin.Function(self.vs[0])
        problem_FEM = dolfin.LinearVariationalProblem(self.a, self.L, T, form_compiler_parameters={"optimize": True})
        solver = dolfin.LinearVariationalSolver(problem_FEM)
        prm = solver.parameters
        prm["linear_solver"] = "gmres"
        prm["preconditioner"] = "amg"
        for i in range(len(self.time_mesh)-1):
            print('solve for t: ', self.time_mesh[i+1])
            self.dt.assign(self.time_mesh[i+1][0]-self.time_mesh[i][0])
            self.q.t = self.time_mesh[i][0]
            # Compute solution
            solver.solve()
            # Update previous solution
            self.T_n.assign(T)

            # store solution
            Ttmp = dolfin.Function(self.vs[0])
            Ttmp.vector()[:] = 1 * T.vector()[:]
            Ttime.append(Ttmp)
            Txfixed.append(np.copy(T(self.pos_fixed)))

        return Ttime, Txfixed  # solution in time over x and time solution at fixed x


# PGD model and Error computation
#=======================================
class problem(unittest.TestCase):

    def setUp(self):

        # define some parameters
        self.param = {"rho": 7100, "c_p": 3100, "k": 100, "r_0": 0.0035, "vel": 0.01, "L": 0.1, "W": 0.1, "H": 0.01,
                      "af": 0.002, "b": 0.002, "c": 0.002, "T_amb": 25, "P": 2500, "t_max": 10, "t_end": 100,
                      "h_min": 4, "h_max": 26}

        self.param.update({"h_g": self.param["af"]*3})

        # self.param.update({'x_ref': 1, 'y_ref': 1, 'z_ref': 1,
        #                     't_ref': 1,
        #                     'r_ref': 1,
        #                     'h_ref': 1, 'T_ref': 1})
        self.param.update({'x_ref': self.param['L'], 'y_ref': self.param['W'], 'z_ref': self.param['H'],
                            't_ref': self.param['t_max']-self.param["r_0"]/self.param["vel"],
                            'r_ref': self.param["vel"]*self.param["t_max"]-self.param["r_0"],
                            'h_ref': self.param['h_max'], 'T_ref': 500})

        # del_dim = self.param['T_ref'] * self.param['rho'] * self.param['c_p']
        del_dim = 1.0
        self.param['a_t'] = self.param['T_ref'] / (self.param['t_ref'] * del_dim)
        self.param['a_r'] = self.param['T_ref'] / (self.param['r_ref'] * del_dim)
        self.param['a_s'] = self.param['T_ref'] / (del_dim)# * self.param['s_ref'])
        self.param['a_s2'] = self.param['T_ref'] / (del_dim)# * self.param['s_ref']**2)
        self.param['a_x2'] = self.param['T_ref'] / (self.param['x_ref']**2 * del_dim)
        self.param['a_y2'] = self.param['T_ref'] / (self.param['y_ref']**2 * del_dim)
        self.param['a_z2'] = self.param['T_ref'] / (self.param['z_ref']**2 * del_dim)
        self.param['a_conv_xy'] = self.param['T_ref'] * self.param['x_ref'] * self.param['y_ref'] / del_dim
        self.param['a_conv_xz'] = self.param['T_ref'] * self.param['x_ref'] * self.param['z_ref'] / del_dim
        self.param['a_conv_yz'] = self.param['T_ref'] * self.param['y_ref'] * self.param['z_ref'] / del_dim
        self.param['l1'] = 1.0 / del_dim
        self.param['l2_xy'] = self.param['x_ref'] * self.param['y_ref'] / (self.param['T_ref'] * del_dim)
        self.param['l2_xz'] = self.param['x_ref'] * self.param['z_ref'] / (self.param['T_ref'] * del_dim)
        self.param['l2_yz'] = self.param['y_ref'] * self.param['z_ref'] / (self.param['T_ref'] * del_dim)

        # global parameters
        self.ord = 1  # order for each mesh
        self.ords = [self.ord, self.ord, self.ord, self.ord, self.ord, self.ord]
        self.ranges_PGD = [[0., 3.],                                                                            # smin, smax
                           [0., self.param["W"] / self.param['y_ref']],                                         # ymin, ymax
                           [0., self.param["H"] / self.param['z_ref']],                                         # zmin, zmax
                           [self.param["r_0"] / self.param['r_ref'],
                            (self.param["vel"]*self.param["t_max"]-self.param["r_0"]) / self.param['r_ref']],   # rmin, rmax
                           [0.5, 1.],                                                                           # etamin, etamax
                           [self.param["h_min"] / self.param["h_ref"],
                            self.param["h_max"] / self.param["h_ref"]]]                                         # hmin, hmax
        self.ranges_FEM = [[0., self.param["L"]],                                         # xmin, xmax
                           [0., self.param["W"]],                                         # ymin, ymax
                           [0., self.param["H"]],                                         # zmin, zmax
                           [(self.param["r_0"]/self.param["vel"]),
                            (self.param["t_max"]-self.param["r_0"]/self.param["vel"])]]   # tmin, tmax
        self.num_elem_PGD = [1000,  # number of elements in s
                             1000,  # number of elements in y
                             100,  # number of elements in z
                             1000,  # number of elements in r
                             100,  # number of elements in eta
                             100]  # number of elements in h
        self.num_elem_FEM = [50,  # number of elements in x
                             50,  # number of elements in y
                             5,  # number of elements in z
                             100]  # number of elements in t

        # evaluation parameters
        self.t_fixed = 0.9 * self.param['t_max'] / self.param['t_ref']
        self.eta_fixed = 1.
        self.h_fixed = 0.5 * (self.param["h_min"]+self.param["h_max"]) / self.param['h_ref']
        self.pos_fixed = [0.5 * self.param['L'] / self.param['x_ref'],
                          0.5 * self.param['W'] / self.param['y_ref'],
                          self.param['H'] / self.param['z_ref']]

        self.plotting = True
        # self.plotting = False

    def TearDown(self):
        pass

    def test_heating(self):
        # case heating
        self.q_coeff = 6 * np.sqrt(3) / ((self.param["af"]+self.param["af"]) * self.param["b"] * self.param["c"] * np.pi**(3/2))

        # q Goldak
        self.q_FEM = dolfin.Expression('x[0] >= vel*t+hg/2 ? 0 : x[0] <= vel*t-hg/2 ? 0 : \
                                        coeff * exp(-3*(pow(x[0]-vel*t,2)/pow(af,2) + pow(x[1]-yc,2)/pow(b,2) + pow(x[2]-zc,2)/pow(c,2)))',
                                       degree=4, coeff=self.q_coeff, af=self.param['af'], b=self.param['b'], c=self.param['c'],
                                       vel=self.param["vel"], hg=self.param["h_g"], yc=self.param['W']/2, zc=self.param['H'], t=0)
        self.q_PGD = [dolfin.Expression('exp(-3*pow((x[0]-1.5)*h_g,2)/pow(af,2))', degree=4, af=self.param["af"], h_g=self.param["h_g"]),
                      dolfin.Expression('exp(-3*pow(x[0]*yref-yc,2)/pow(b,2))', degree=4, b=self.param["b"], yc=self.param["W"]/2, yref=self.param['y_ref']),
                      dolfin.Expression('exp(-3*pow(x[0]*zref-zc,2)/pow(c,2))', degree=4, c=self.param["c"], zc=self.param["H"], zref=self.param['z_ref'])]

        self.param['Tamb_fct'] = dolfin.Expression('Tamb', degree=1, Tamb=self.param["T_amb"] / self.param['T_ref']) # initial condition FEM
        self.param['IC_t'] = self.param['Tamb_fct']
        self.param['IC_x'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_y'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_z'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_eta'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_r'] = self.param['Tamb_fct']
        self.param['IC_s'] = dolfin.Expression('1.0', degree=1)
        self.param['IC_h'] = dolfin.Expression('1.0', degree=1)

        # MESH
        meshes_FEM, vs_FEM = create_meshes_FEM(self.num_elem_FEM, self.ords, self.ranges_FEM)
        meshes_PGD, vs_PGD = create_meshes_PGD(self.num_elem_PGD, self.ords, self.ranges_PGD)

        # PGD
        pgd_fd, param = create_PGD(param=self.param, vs=vs_PGD, q_PGD=self.q_PGD, q_coeff=self.q_coeff)

        # cooling
        from thermo_3D_dimless_convection_cooling import PgdCooling
        pgd_cooling, param_cool = PgdCooling(pgd_fd, self.param)(self.pos_fixed, self.t_fixed, self.eta_fixed, self.h_fixed)
        cool_sol = pgd_cooling.evaluate(3, [0, 1, 2, 4, 5], [self.pos_fixed[0], self.pos_fixed[1], self.pos_fixed[2], self.eta_fixed, self.h_fixed], 0)

        # FEM reference solution
        u_fem, u_fem2 = Reference(param=self.param, vs=vs_FEM, q=self.q_FEM, pos_fixed=self.pos_fixed)(
            [self.t_fixed, self.eta_fixed, self.h_fixed])

        # PGD solution at fixed place over time
        upgd_fd2, time_PGD = remapping(pgd_fd, self.param, self.pos_fixed, self.eta_fixed, self.h_fixed)

        # error computation
        errors_FEM = np.linalg.norm(np.interp(meshes_FEM[1].coordinates()[:],
                                              time_PGD * self.param['r_ref'], upgd_fd2 * self.param['T_ref']) - u_fem2) / np.linalg.norm(u_fem2)  # PGD FD - FEM
        print('rel error over time: ', errors_FEM)

        if self.plotting:
            #### plotting optional
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(meshes_FEM[1].coordinates()[:], np.array(u_fem2), '-og', label='FEM')
            plt.plot(time_PGD * self.param['r_ref'], upgd_fd2 * self.param['T_ref'], '-*r', label='PGD FD')
            plt.title(f"PGD solution at [x,eta,h]={self.pos_fixed[0]*self.param['x_ref']},{self.eta_fixed},{self.h_fixed*self.param['h_ref']} over time")
            plt.xlabel("time t [s]")
            plt.ylabel("Temperature T [°C]")
            plt.legend()
            plt.draw()
            plt.show()

            plt.figure()
            plt.plot(pgd_cooling.mesh[3].dataX[:] * param_cool['t_ref_cool'], cool_sol * param_cool['T_ref'], '-*b', label='PGD FD')
            plt.title('cooling phase')
            plt.xlabel("time t [s]")
            plt.ylabel("Temperature T [°C]")
            plt.legend()
            plt.draw()
            plt.show()

        # self.assertTrue(errors_FEM < 1e-2)

if __name__ == '__main__':
    dolfin.set_log_level(dolfin.LogLevel.ERROR)

    import logging
    logging.basicConfig(level=logging.INFO)

    unittest.main()
