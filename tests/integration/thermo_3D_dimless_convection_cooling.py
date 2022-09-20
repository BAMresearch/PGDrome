import dolfin
import numpy as np

from pgdrome.solver import PGDProblem1, FD_matrices


def create_meshes(num_elem, ord, ranges, old_vs, refinement_t=None):
    '''
    :param num_elem: list for each PG CO
    :param ord: list for each PG CO
    :param ranges: list for each PG CO
    :return: meshes and V
    '''

    print('create meshes PGD cooling')

    vs = old_vs
    mapped_dim = [0, 3]

    refinement_methods = {  # dictionary to decide which refinement is used
        "cos": "cos"
    }

    for i in mapped_dim:
        if i == 3:
            if refinement_t is None:
                mesh_tmp = dolfin.IntervalMesh(num_elem[i], ranges[i][0], ranges[i][1])
            elif refinement_t == refinement_methods["cos"]:
                Nt = num_elem[i]
                lt = ranges[i][1] - ranges[i][0]
                mesh_tmp = dolfin.UnitIntervalMesh(Nt)
                x = mesh_tmp.coordinates()
                x[:] = x[:] / 2
                x[:] = (x[:] - 0.5) * 2.
                x[:] = 0.5 * (np.cos(np.pi * (x[:] - 1.) / 2.) + 1.)
                x[:] = x[:] * lt * 2 + ranges[i][0]
            else:
                raise NotImplementedError('Such a refinement is not implemented!')
        else:
            mesh_tmp = dolfin.IntervalMesh(num_elem[i], ranges[i][0], ranges[i][1])
        vs_tmp = dolfin.FunctionSpace(mesh_tmp, 'CG', ord[i])

        vs[i] = vs_tmp

    return vs


def create_dom(vs, param):
    # create boundarydomains
    boundarydomain_x = dolfin.MeshFunction("size_t", vs[0].mesh(), vs[0].mesh().topology().dim() - 1)
    boundarydomain_x.set_all(0)
    boundarydomain_y = dolfin.MeshFunction("size_t", vs[1].mesh(), vs[1].mesh().topology().dim() - 1)
    boundarydomain_y.set_all(0)
    boundarydomain_z = dolfin.MeshFunction("size_t", vs[2].mesh(), vs[2].mesh().topology().dim() - 1)
    boundarydomain_z.set_all(0)

    class left_tight(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return np.isclose(x, np.min(vs[0].mesh().coordinates()[:])) or np.isclose(x, np.max(
                vs[0].mesh().coordinates()[:]))

    class front_back(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return np.isclose(x, np.min(vs[1].mesh().coordinates()[:])) or np.isclose(x, np.max(
                vs[1].mesh().coordinates()[:]))

    class top_bottom(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return np.isclose(x, np.max(vs[2].mesh().coordinates()[:])) or np.isclose(x, np.min(
                vs[2].mesh().coordinates()[:]))

    Left_tight_dom = left_tight()
    Left_tight_dom.mark(boundarydomain_x, 1)
    Front_Back_dom = front_back()
    Front_Back_dom.mark(boundarydomain_y, 1)
    Top_Bottom_dom = top_bottom()
    Top_Bottom_dom.mark(boundarydomain_z, 1)

    dom = [boundarydomain_x, boundarydomain_y, boundarydomain_z]

    return dom


def create_bc(vs, dom, param):
    # Initial condition
    def init(x, on_boundary):
        return x < np.min(vs[3].mesh().coordinates()[:]) + 1E-8

    initCond = dolfin.DirichletBC(vs[3], dolfin.Constant(0.), init)

    return [0, 0, 0, initCond, 0, 0]  # x,y,z,t,eta,h


def remap_modes(pgd_heating=None, vs=[], param=None):
    # ==================================================================== #
    #                           SETUP VARIABLES                            #
    # ==================================================================== #

    # get modes from PGD solution
    PGD_modes = list()
    num_modes = pgd_heating.numModes
    for dim in range(len(vs)):
        temp = list()
        for i in range(num_modes):
            temp.append(pgd_heating.mesh[dim].attributes[0].interpolationfct[i])
        PGD_modes.append(temp)

    # define mapping parameters
    h_g = param['h_g']
    h_1 = param['h_1']
    h_2 = param['h_2']

    r_last = np.max(pgd_heating.mesh[3].dataX[:])

    x_coords = vs[0].mesh().coordinates()

    # ==================================================================== #
    #                           COMPUTE MAPPING                            #
    # ==================================================================== #

    funcs = list()
    for dim in range(len(vs)):
        funcs.append(list())

    for m_idx in range(num_modes):
        x_mode = np.zeros(len(x_coords))
        for i, x in enumerate(x_coords):
            # left side
            s_temp = x * param['x_ref'] / h_1(r_last)
            if s_temp <= 1:
                u_x = PGD_modes[0][m_idx](s_temp)
            else:
                # center
                s_temp = (x * param['x_ref'] - h_1(r_last)) / h_g + 1
                if 1 < s_temp <= 2:
                    u_x = PGD_modes[0][m_idx](s_temp)
                else:
                    # right side
                    s_temp = (x * param['x_ref'] - h_1(r_last) - h_g) / h_2(r_last) + 2
                    u_x = PGD_modes[0][m_idx](s_temp - 1E-8)

            x_mode[i] = u_x

        for dim in range(len(vs)):
            funcs[dim].append(dolfin.Function(vs[dim]))
            if dim == 0:
                funcs[dim][m_idx].vector()[:] = x_mode[dolfin.dof_to_vertex_map(vs[dim])]
            elif dim == 3:
                # tmp = np.zeros(len(vs[dim].mesh().coordinates()))
                # tmp[-1] = PGD_modes[dim][m_idx](r_last)
                # funcs[dim][m_idx].vector()[:] = tmp
                funcs[dim][m_idx].vector()[:] = np.ones(len(vs[dim].mesh().coordinates())) * PGD_modes[dim][m_idx](
                    r_last)

            else:
                if len(pgd_heating.mesh[dim].dataX[:]) == len(vs[dim].mesh().coordinates()):
                    funcs[dim][m_idx].vector()[:] = PGD_modes[dim][m_idx].compute_vertex_values()[
                        dolfin.dof_to_vertex_map(vs[dim])]
                else:
                    funcs[dim][m_idx] = dolfin.interpolate(PGD_modes[dim][m_idx], vs[dim])

    IC = [param["IC_x"], param["IC_y"], param["IC_z"], param["IC_t"], param["IC_eta"], param["IC_h"]]
    for dim in range(len(vs)):
        # add heating initial condition, which is missing in the modes
        funcs[dim].append(dolfin.interpolate(IC[dim], vs[dim]))
        # override old functions with functions regarding the actual vs
        PGD_modes[dim] = funcs[dim]

    return PGD_modes


def problem_assemble_lhs(fct_F, var_F, Fs, meshes, dom, param, typ, dim):
    # define measures
    ds_x = dolfin.Measure('ds', domain=meshes[0], subdomain_data=dom[0])
    ds_y = dolfin.Measure('ds', domain=meshes[1], subdomain_data=dom[1])
    ds_z = dolfin.Measure('ds', domain=meshes[2], subdomain_data=dom[2])

    alpha_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ Fs[3].vector()[:]
    alpha_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ Fs[3].vector()[:]
    alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ Fs[4].vector()[:]
    alpha_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ Fs[5].vector()[:]
    alpha_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ Fs[5].vector()[:]

    if typ == 'x':
        a = dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                            * alpha_t2
                            * alpha_eta1
                            * alpha_h1) \
            * param['a_t_cool'] * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                              * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h1) \
            * param['a_x2'] * param["k"] * var_F.dx(0) * fct_F.dx(0) * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1]))
                              * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h1) \
            * param['a_y2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                              * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h1) \
            * param['a_z2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                              * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h2) \
            * param['a_conv_yz'] * var_F * fct_F * ds_x \
            + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * ds_y)
                              * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h2) \
            * param['a_conv_xz'] * var_F * fct_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                              * dolfin.assemble(Fs[2] * Fs[2] * ds_z)
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h2) \
            * param['a_conv_xy'] * var_F * fct_F * dolfin.dx(meshes[0])

    if typ == 'y':
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                            * alpha_t2
                            * alpha_eta1
                            * alpha_h1) \
            * param['a_t_cool'] * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h1) \
            * param['a_x2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h1) \
            * param['a_y2'] * param["k"] * var_F.dx(0) * fct_F.dx(0) * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h1) \
            * param['a_z2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * ds_x)
                              * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h2) \
            * param['a_conv_yz'] * var_F * fct_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h2) \
            * param['a_conv_xz'] * var_F * fct_F * ds_y \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[2] * Fs[2] * ds_z)
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h2) \
            * param['a_conv_xy'] * var_F * fct_F * dolfin.dx(meshes[1])

    if typ == 'z':
        a = dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                            * alpha_t2
                            * alpha_eta1
                            * alpha_h1) \
            * param['a_t_cool'] * param["rho"] * param["c_p"] * var_F * fct_F * dolfin.dx(meshes[2]) \
            + dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h1) \
            * param['a_x2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[2]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h1) \
            * param['a_y2'] * param["k"] * var_F * fct_F * dolfin.dx(meshes[2]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h1) \
            * param['a_z2'] * param["k"] * var_F.dx(0) * fct_F.dx(0) * dolfin.dx(meshes[2]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * ds_x)
                              * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h2) \
            * param['a_conv_yz'] * var_F * fct_F * dolfin.dx(meshes[2]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[1] * Fs[1] * ds_y)
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h2) \
            * param['a_conv_xz'] * var_F * fct_F * dolfin.dx(meshes[2]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
                              * alpha_t1
                              * alpha_eta1
                              * alpha_h2) \
            * param['a_conv_xy'] * var_F * fct_F * ds_z

    if typ == 't':
        a = dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_eta1 \
            * alpha_h1 \
            * param['a_t_cool'] * param["rho"] * param["c_p"] * param['D1_up_t'] \
            + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_eta1 \
            * alpha_h1 \
            * param['a_x2'] * param["k"] * param['M_t'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_eta1 \
            * alpha_h1 \
            * param['a_y2'] * param["k"] * param['M_t'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
            * alpha_eta1 \
            * alpha_h1 \
            * param['a_z2'] * param["k"] * param['M_t'] \
            + dolfin.assemble(Fs[0] * Fs[0] * ds_x) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_eta1 \
            * alpha_h2 \
            * param['a_conv_yz'] * param['M_t'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * ds_y) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_eta1 \
            * alpha_h2 \
            * param['a_conv_xz'] * param['M_t'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * ds_z) \
            * alpha_eta1 \
            * alpha_h2 \
            * param['a_conv_xy'] * param['M_t']

        # add initial condition
        a[:, param['bc_idx']] = 0
        a[param['bc_idx'], :] = 0
        a[param['bc_idx'], param['bc_idx']] = 1

    if typ == 'eta':
        a = dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t2 \
            * alpha_h1 \
            * param['a_t_cool'] * param["rho"] * param["c_p"] * param['M_eta'] \
            + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_h1 \
            * param['a_x2'] * param["k"] * param['M_eta'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_h1 \
            * param['a_y2'] * param["k"] * param['M_eta'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_h1 \
            * param['a_z2'] * param["k"] * param['M_eta'] \
            + dolfin.assemble(Fs[0] * Fs[0] * ds_x) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_h2 \
            * param['a_conv_yz'] * param['M_eta'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * ds_y) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_h2 \
            * param['a_conv_xz'] * param['M_eta'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * ds_z) \
            * alpha_t1 \
            * alpha_h2 \
            * param['a_conv_xy'] * param['M_eta']

    if typ == 'h':
        a = dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t2 \
            * alpha_eta1 \
            * param['a_t_cool'] * param["rho"] * param["c_p"] * param['M_h'] \
            + dolfin.assemble(Fs[0].dx(0) * Fs[0].dx(0) * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_eta1 \
            * param['a_x2'] * param["k"] * param['M_h'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1].dx(0) * Fs[1].dx(0) * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_eta1 \
            * param['a_y2'] * param["k"] * param['M_h'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2].dx(0) * Fs[2].dx(0) * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_eta1 \
            * param['a_z2'] * param["k"] * param['M_h'] \
            + dolfin.assemble(Fs[0] * Fs[0] * ds_x) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_eta1 \
            * param['a_conv_yz'] * param['h'].vector()[:] * param['M_h'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * ds_y) \
            * dolfin.assemble(Fs[2] * Fs[2] * dolfin.dx(meshes[2])) \
            * alpha_t1 \
            * alpha_eta1 \
            * param['a_conv_xz'] * param['h'].vector()[:] * param['M_h'] \
            + dolfin.assemble(Fs[0] * Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * Fs[2] * ds_z) \
            * alpha_t1 \
            * alpha_eta1 \
            * param['a_conv_xy'] * param['h'].vector()[:] * param['M_h']

    return a


def problem_assemble_rhs(fct_F, var_F, Fs, meshes, dom, param, Q, PGD_func, typ, nE, dim):
    # problem description right hand side of DGL for each fixed point problem

    # define measures
    ds_x = dolfin.Measure('ds', domain=meshes[0], subdomain_data=dom[0])
    ds_y = dolfin.Measure('ds', domain=meshes[1], subdomain_data=dom[1])
    ds_z = dolfin.Measure('ds', domain=meshes[2], subdomain_data=dom[2])

    IC = [param["IC_x_old"], param["IC_y_old"], param["IC_z_old"], param["IC_t_old"], param["IC_eta_old"],
          param["IC_h_old"]]

    num_old_modes = len(IC[0])

    beta_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ np.ones(len(Fs[3].vector()))
    beta_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ np.ones(len(Fs[4].vector()))
    beta_h1 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ np.ones(len(Fs[5].vector()))

    if typ == 'x':
        l = dolfin.Constant(dolfin.assemble(Fs[1] * dolfin.dx(meshes[1]))
                            * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2]))
                            * beta_t1
                            * beta_eta1
                            * beta_h1) \
            * param['l2_yz'] * param["T_amb"] * var_F * ds_x \
            + dolfin.Constant(dolfin.assemble(Fs[1] * ds_y)
                              * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2]))
                              * beta_t1
                              * beta_eta1
                              * beta_h1) \
            * param['l2_xz'] * param["T_amb"] * var_F * dolfin.dx(meshes[0]) \
            + dolfin.Constant(dolfin.assemble(Fs[1] * dolfin.dx(meshes[1]))
                              * dolfin.assemble(Fs[2] * ds_z)
                              * beta_t1
                              * beta_eta1
                              * beta_h1) \
            * param['l2_xy'] * param["T_amb"] * var_F * dolfin.dx(meshes[0])
        for m_idx in range(num_old_modes):
            alpha_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ IC[3][m_idx].vector()[:]
            alpha_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ IC[3][m_idx].vector()[:]
            alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ IC[4][m_idx].vector()[:]
            alpha_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ IC[5][m_idx].vector()[:]
            alpha_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ IC[5][m_idx].vector()[
                                                                                                 :]
            l += - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t2
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_t_cool'] * param["rho"] * param["c_p"] * var_F * IC[0][m_idx] * dolfin.dx(meshes[0]) \
                 - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_x2'] * param["k"] * var_F.dx(0) * IC[0][m_idx].dx(0) * dolfin.dx(meshes[0]) \
                 - dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * IC[1][m_idx].dx(0) * dolfin.dx(meshes[1]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_y2'] * param["k"] * var_F * IC[0][m_idx] * dolfin.dx(meshes[0]) \
                 - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * dolfin.assemble(Fs[2].dx(0) * IC[2][m_idx].dx(0) * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_z2'] * param["k"] * var_F * IC[0][m_idx] * dolfin.dx(meshes[0]) \
                 - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h2) \
                 * param['a_conv_yz'] * var_F * IC[0][m_idx] * ds_x \
                 - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1][m_idx] * ds_y)
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h2) \
                 * param['a_conv_xz'] * var_F * IC[0][m_idx] * dolfin.dx(meshes[0]) \
                 - dolfin.Constant(dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * ds_z)
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h2) \
                 * param['a_conv_xy'] * var_F * IC[0][m_idx] * dolfin.dx(meshes[0])
        if nE > 0:
            for old in range(nE):
                alpha_old_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ \
                               PGD_func[5][old].vector()[:]

                l += - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t2
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_t_cool'] * param["rho"] * param["c_p"] * var_F * PGD_func[0][old] * dolfin.dx(meshes[0]) \
                     - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_x2'] * param["k"] * var_F.dx(0) * PGD_func[0][old].dx(0) * dolfin.dx(meshes[0]) \
                     - dolfin.Constant(dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_y2'] * param["k"] * var_F * PGD_func[0][old] * dolfin.dx(meshes[0]) \
                     - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_z2'] * param["k"] * var_F * PGD_func[0][old] * dolfin.dx(meshes[0]) \
                     - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h2) \
                     * param['a_conv_yz'] * var_F * PGD_func[0][old] * ds_x \
                     - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y)
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h2) \
                     * param['a_conv_xz'] * var_F * PGD_func[0][old] * dolfin.dx(meshes[0]) \
                     - dolfin.Constant(dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z)
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h2) \
                     * param['a_conv_xy'] * var_F * PGD_func[0][old] * dolfin.dx(meshes[0])

    if typ == 'y':
        l = dolfin.Constant(dolfin.assemble(Fs[0] * ds_x)
                            * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2]))
                            * beta_t1
                            * beta_eta1
                            * beta_h1) \
            * param['l2_yz'] * param["T_amb"] * var_F * dolfin.dx(meshes[1]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2]))
                              * beta_t1
                              * beta_eta1
                              * beta_h1) \
            * param['l2_xz'] * param["T_amb"] * var_F * ds_y \
            + dolfin.Constant(dolfin.assemble(Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[2] * ds_z)
                              * beta_t1
                              * beta_eta1
                              * beta_h1) \
            * param['l2_xy'] * param["T_amb"] * var_F * dolfin.dx(meshes[1])
        for m_idx in range(num_old_modes):
            alpha_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ IC[3][m_idx].vector()[:]
            alpha_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ IC[3][m_idx].vector()[:]
            alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ IC[4][m_idx].vector()[:]
            alpha_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ IC[5][m_idx].vector()[:]
            alpha_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ IC[5][m_idx].vector()[
                                                                                                 :]
            l += - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t2
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_t_cool'] * param["rho"] * param["c_p"] * var_F * IC[1][m_idx] * dolfin.dx(meshes[1]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * IC[0][m_idx].dx(0) * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_x2'] * param["k"] * var_F * IC[1][m_idx] * dolfin.dx(meshes[1]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_y2'] * param["k"] * var_F.dx(0) * IC[1][m_idx].dx(0) * dolfin.dx(meshes[1]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[2].dx(0) * IC[2][m_idx].dx(0) * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_z2'] * param["k"] * var_F * IC[1][m_idx] * dolfin.dx(meshes[1]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * ds_x)
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h2) \
                 * param['a_conv_yz'] * var_F * IC[1][m_idx] * dolfin.dx(meshes[1]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h2) \
                 * param['a_conv_xz'] * var_F * IC[1][m_idx] * ds_y \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[2] * IC[2][m_idx] * ds_z)
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h2) \
                 * param['a_conv_xy'] * var_F * IC[1][m_idx] * dolfin.dx(meshes[1])
        if nE > 0:
            for old in range(nE):
                alpha_old_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ \
                               PGD_func[5][old].vector()[:]

                l += - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t2
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_t_cool'] * param["rho"] * param["c_p"] * var_F * PGD_func[1][old] * dolfin.dx(
                    meshes[1]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_x2'] * param["k"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_y2'] * param["k"] * var_F.dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(
                    Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_z2'] * param["k"] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_x)
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h2) \
                     * param['a_conv_yz'] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h2) \
                     * param['a_conv_xz'] * var_F * PGD_func[1][old] * ds_y \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z)
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h2) \
                     * param['a_conv_xy'] * var_F * PGD_func[1][old] * dolfin.dx(meshes[1])

    if typ == 'z':
        l = dolfin.Constant(dolfin.assemble(Fs[0] * ds_x)
                            * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1]))
                            * beta_t1
                            * beta_eta1
                            * beta_h1) \
            * param['l2_yz'] * param["T_amb"] * var_F * dolfin.dx(meshes[2]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[1] * ds_y)
                              * beta_t1
                              * beta_eta1
                              * beta_h1) \
            * param['l2_xz'] * param["T_amb"] * var_F * dolfin.dx(meshes[2]) \
            + dolfin.Constant(dolfin.assemble(Fs[0] * dolfin.dx(meshes[0]))
                              * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1]))
                              * beta_t1
                              * beta_eta1
                              * beta_h1) \
            * param['l2_xy'] * param["T_amb"] * var_F * ds_z
        for m_idx in range(num_old_modes):
            alpha_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ IC[3][m_idx].vector()[:]
            alpha_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ IC[3][m_idx].vector()[:]
            alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ IC[4][m_idx].vector()[:]
            alpha_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ IC[5][m_idx].vector()[:]
            alpha_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ IC[5][m_idx].vector()[
                                                                                                 :]
            l += - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * alpha_t2
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_t_cool'] * param["rho"] * param["c_p"] * var_F * IC[2][m_idx] * dolfin.dx(meshes[2]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * IC[0][m_idx].dx(0) * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_x2'] * param["k"] * var_F * IC[2][m_idx] * dolfin.dx(meshes[2]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[1].dx(0) * IC[1][m_idx].dx(0) * dolfin.dx(meshes[1]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_y2'] * param["k"] * var_F * IC[2][m_idx] * dolfin.dx(meshes[2]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h1) \
                 * param['a_z2'] * param["k"] * var_F.dx(0) * IC[2][m_idx].dx(0) * dolfin.dx(meshes[2]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * ds_x)
                                   * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h2) \
                 * param['a_conv_yz'] * var_F * IC[2][m_idx] * dolfin.dx(meshes[2]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[1] * IC[1][m_idx] * ds_y)
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h2) \
                 * param['a_conv_xz'] * var_F * IC[2][m_idx] * dolfin.dx(meshes[2]) \
                 - dolfin.Constant(dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0]))
                                   * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1]))
                                   * alpha_t1
                                   * alpha_eta1
                                   * alpha_h2) \
                 * param['a_conv_xy'] * var_F * IC[2][m_idx] * ds_z
        if nE > 0:
            for old in range(nE):
                alpha_old_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ \
                               PGD_func[5][old].vector()[:]

                l += - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * alpha_old_t2
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_t_cool'] * param["rho"] * param["c_p"] * var_F * PGD_func[2][old] * dolfin.dx(
                    meshes[2]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_x2'] * param["k"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(
                    Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_y2'] * param["k"] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h1) \
                     * param['a_z2'] * param["k"] * var_F.dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_x)
                                       * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h2) \
                     * param['a_conv_yz'] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y)
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h2) \
                     * param['a_conv_xz'] * var_F * PGD_func[2][old] * dolfin.dx(meshes[2]) \
                     - dolfin.Constant(dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0]))
                                       * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1]))
                                       * alpha_old_t1
                                       * alpha_old_eta1
                                       * alpha_old_h2) \
                     * param['a_conv_xy'] * var_F * PGD_func[2][old] * ds_z

    if typ == 't':
        l = dolfin.assemble(Fs[0] * ds_x) \
            * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
            * beta_eta1 \
            * beta_h1 \
            * param['l2_yz'] * param["T_amb"] * param['M_t'] @ np.ones(len(Fs[3].vector())) \
            + dolfin.assemble(Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * ds_y) \
            * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
            * beta_eta1 \
            * beta_h1 \
            * param['l2_xz'] * param["T_amb"] * param['M_t'] @ np.ones(
            len(Fs[3].vector())) \
            + dolfin.assemble(Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * ds_z) \
            * beta_eta1 \
            * beta_h1 \
            * param['l2_xy'] * param["T_amb"] * param['M_t'] @ np.ones(
            len(Fs[3].vector()))
        for m_idx in range(num_old_modes):
            alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ IC[4][m_idx].vector()[:]
            alpha_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ IC[5][m_idx].vector()[:]
            alpha_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ IC[5][m_idx].vector()[
                                                                                                 :]
            l += - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_eta1 \
                 * alpha_h1 \
                 * param['a_t_cool'] * param["rho"] * param["c_p"] * param['D1_up_t'] @ IC[3][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0].dx(0) * IC[0][m_idx].dx(0) * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_eta1 \
                 * alpha_h1 \
                 * param['a_x2'] * param["k"] * param['M_t'] @ IC[3][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1].dx(0) * IC[1][m_idx].dx(0) * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_eta1 \
                 * alpha_h1 \
                 * param['a_y2'] * param["k"] * param['M_t'] @ IC[3][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2].dx(0) * IC[2][m_idx].dx(0) * dolfin.dx(meshes[2])) \
                 * alpha_eta1 \
                 * alpha_h1 \
                 * param['a_z2'] * param["k"] * param['M_t'] @ IC[3][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * ds_x) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_eta1 \
                 * alpha_h2 \
                 * param['a_conv_yz'] * param['M_t'] @ IC[3][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * ds_y) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_eta1 \
                 * alpha_h2 \
                 * param['a_conv_xz'] * param['M_t'] @ IC[3][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * ds_z) \
                 * alpha_eta1 \
                 * alpha_h2 \
                 * param['a_conv_xy'] * param['M_t'] @ IC[3][m_idx].vector()[:]
        if nE > 0:
            for old in range(nE):
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ \
                               PGD_func[5][old].vector()[:]

                l += - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_eta1 \
                     * alpha_old_h1 \
                     * param['a_t_cool'] * param["rho"] * param["c_p"] * param['D1_up_t'] @ PGD_func[3][
                                                                                                old].vector()[
                                                                                            :] \
                     - dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_eta1 \
                     * alpha_old_h1 \
                     * param['a_x2'] * param["k"] * param['M_t'] @ PGD_func[3][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_eta1 \
                     * alpha_old_h1 \
                     * param['a_y2'] * param["k"] * param['M_t'] @ PGD_func[3][
                                                                       old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                     * alpha_old_eta1 \
                     * alpha_old_h1 \
                     * param['a_z2'] * param["k"] * param['M_t'] @ PGD_func[3][
                                                                       old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_x) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_eta1 \
                     * alpha_old_h2 \
                     * param['a_conv_yz'] * param['M_t'] @ PGD_func[3][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_eta1 \
                     * alpha_old_h2 \
                     * param['a_conv_xz'] * param['M_t'] @ PGD_func[3][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z) \
                     * alpha_old_eta1 \
                     * alpha_old_h2 \
                     * param['a_conv_xy'] * param['M_t'] @ PGD_func[3][old].vector()[:]

        # add initial condition
        l[param['bc_idx']] = 0

    if typ == 'eta':
        l = dolfin.assemble(Fs[0] * ds_x) \
            * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
            * beta_t1 \
            * beta_h1 \
            * param['l2_yz'] * param["T_amb"] * param['M_eta'] @ np.ones(len(Fs[4].vector())) \
            + dolfin.assemble(Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * ds_y) \
            * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
            * beta_t1 \
            * beta_h1 \
            * param['l2_xz'] * param["T_amb"] * param['M_eta'] @ np.ones(len(Fs[4].vector())) \
            + dolfin.assemble(Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * ds_z) \
            * beta_t1 \
            * beta_h1 \
            * param['l2_xy'] * param["T_amb"] * param['M_eta'] @ np.ones(len(Fs[4].vector()))
        for m_idx in range(num_old_modes):
            alpha_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ IC[3][m_idx].vector()[:]
            alpha_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ IC[3][m_idx].vector()[:]
            alpha_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ IC[5][m_idx].vector()[:]
            alpha_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ IC[5][m_idx].vector()[
                                                                                                 :]
            l += - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t2 \
                 * alpha_h1 \
                 * param['a_t_cool'] * param["rho"] * param["c_p"] * param['M_eta'] @ IC[4][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0].dx(0) * IC[0][m_idx].dx(0) * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_h1 \
                 * param['a_x2'] * param["k"] * param['M_eta'] @ IC[4][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1].dx(0) * IC[1][m_idx].dx(0) * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_h1 \
                 * param['a_y2'] * param["k"] * param['M_eta'] @ IC[4][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2].dx(0) * IC[2][m_idx].dx(0) * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_h1 \
                 * param['a_z2'] * param["k"] * param['M_eta'] @ IC[4][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * ds_x) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_h2 \
                 * param['a_conv_yz'] * param['M_eta'] @ IC[4][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * ds_y) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_h2 \
                 * param['a_conv_xz'] * param['M_eta'] @ IC[4][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * ds_z) \
                 * alpha_t1 \
                 * alpha_h2 \
                 * param['a_conv_xy'] * param['M_eta'] @ IC[4][m_idx].vector()[:]
        if nE > 0:
            for old in range(nE):
                alpha_old_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ PGD_func[5][old].vector()[:]
                alpha_old_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ \
                               PGD_func[5][old].vector()[:]

                l += - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t2 \
                     * alpha_old_h1 \
                     * param['a_t_cool'] * param["rho"] * param["c_p"] * param['M_eta'] @ PGD_func[4][old].vector()[
                                                                                          :] \
                     - dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_h1 \
                     * param['a_x2'] * param["k"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_h1 \
                     * param['a_y2'] * param["k"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_h1 \
                     * param['a_z2'] * param["k"] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_x) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_h2 \
                     * param['a_conv_yz'] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_h2 \
                     * param['a_conv_xz'] * param['M_eta'] @ PGD_func[4][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z) \
                     * alpha_old_t1 \
                     * alpha_old_h2 \
                     * param['a_conv_xy'] * param['M_eta'] @ PGD_func[4][old].vector()[:]

    if typ == 'h':
        l = dolfin.assemble(Fs[0] * ds_x) \
            * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
            * beta_t1 \
            * beta_eta1 \
            * param['l2_yz'] * param["T_amb"] * param['h'].vector()[:] * param['M_h'] @ np.ones(
            len(Fs[5].vector())) \
            + dolfin.assemble(Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * ds_y) \
            * dolfin.assemble(Fs[2] * dolfin.dx(meshes[2])) \
            * beta_t1 \
            * beta_eta1 \
            * param['l2_xz'] * param["T_amb"] * param['h'].vector()[:] * param['M_h'] @ np.ones(
            len(Fs[5].vector())) \
            + dolfin.assemble(Fs[0] * dolfin.dx(meshes[0])) \
            * dolfin.assemble(Fs[1] * dolfin.dx(meshes[1])) \
            * dolfin.assemble(Fs[2] * ds_z) \
            * beta_t1 \
            * beta_eta1 \
            * param['l2_xy'] * param["T_amb"] * param['h'].vector()[:] * param['M_h'] @ np.ones(
            len(Fs[5].vector()))
        for m_idx in range(num_old_modes):
            alpha_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ IC[3][m_idx].vector()[:]
            alpha_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ IC[3][m_idx].vector()[:]
            alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ IC[4][m_idx].vector()[:]

            l += - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t2 \
                 * alpha_eta1 \
                 * param['a_t_cool'] * param["rho"] * param["c_p"] * param['M_h'] @ IC[5][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0].dx(0) * IC[0][m_idx].dx(0) * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_eta1 \
                 * param['a_x2'] * param["k"] * param['M_h'] @ IC[5][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1].dx(0) * IC[1][m_idx].dx(0) * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_eta1 \
                 * param['a_y2'] * param["k"] * param['M_h'] @ IC[5][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2].dx(0) * IC[2][m_idx].dx(0) * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_eta1 \
                 * param['a_z2'] * param["k"] * param['M_h'] @ IC[5][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * ds_x) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_eta1 \
                 * param['a_conv_yz'] * param['h'].vector()[:] * param['M_h'] @ IC[5][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * ds_y) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * dolfin.dx(meshes[2])) \
                 * alpha_t1 \
                 * alpha_eta1 \
                 * param['a_conv_xz'] * param['h'].vector()[:] * param['M_h'] @ IC[5][m_idx].vector()[:] \
                 - dolfin.assemble(Fs[0] * IC[0][m_idx] * dolfin.dx(meshes[0])) \
                 * dolfin.assemble(Fs[1] * IC[1][m_idx] * dolfin.dx(meshes[1])) \
                 * dolfin.assemble(Fs[2] * IC[2][m_idx] * ds_z) \
                 * alpha_t1 \
                 * alpha_eta1 \
                 * param['a_conv_xy'] * param['h'].vector()[:] * param['M_h'] @ IC[5][m_idx].vector()[:]
        if nE > 0:
            for old in range(nE):
                alpha_old_t1 = Fs[3].vector()[:].transpose() @ param['M_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_t2 = Fs[3].vector()[:].transpose() @ param['D1_up_t'] @ PGD_func[3][old].vector()[:]
                alpha_old_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ PGD_func[4][old].vector()[:]

                l += - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t2 \
                     * alpha_old_eta1 \
                     * param['a_t_cool'] * param["rho"] * param["c_p"] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                     - dolfin.assemble(Fs[0].dx(0) * PGD_func[0][old].dx(0) * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_eta1 \
                     * param['a_x2'] * param["k"] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1].dx(0) * PGD_func[1][old].dx(0) * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_eta1 \
                     * param['a_y2'] * param["k"] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2].dx(0) * PGD_func[2][old].dx(0) * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_eta1 \
                     * param['a_z2'] * param["k"] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * ds_x) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_eta1 \
                     * param['a_conv_yz'] * param['h'].vector()[:] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * ds_y) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * dolfin.dx(meshes[2])) \
                     * alpha_old_t1 \
                     * alpha_old_eta1 \
                     * param['a_conv_xz'] * param['h'].vector()[:] * param['M_h'] @ PGD_func[5][old].vector()[:] \
                     - dolfin.assemble(Fs[0] * PGD_func[0][old] * dolfin.dx(meshes[0])) \
                     * dolfin.assemble(Fs[1] * PGD_func[1][old] * dolfin.dx(meshes[1])) \
                     * dolfin.assemble(Fs[2] * PGD_func[2][old] * ds_z) \
                     * alpha_old_t1 \
                     * alpha_old_eta1 \
                     * param['a_conv_xy'] * param['h'].vector()[:] * param['M_h'] @ PGD_func[5][old].vector()[:]

    return l


class PgdCooling:

    def __init__(self, pgd_heating=None, param={}, vs_heating=[]):
        self.pgd_solution = pgd_heating
        self.param = param

        self.ord = 1  # order for each mesh
        self.ords = [self.ord, self.ord, self.ord, self.ord, self.ord, self.ord]

        # self.param.update({'t_ref_cool': 1})
        self.param.update({'t_ref_cool': self.param['t_end']})
        self.param.update({'a_t_cool': self.param['a_t'] * self.param['t_ref'] / self.param['t_ref_cool']})

        self.ranges = [[0., self.param["L"] / self.param['x_ref']],
                       [0., self.param["W"] / self.param['y_ref']],
                       [0., self.param["H"] / self.param['z_ref']],
                       [(self.pgd_solution.mesh[3].dataX[-1] * self.param['r_ref'] / self.param['vel']) / self.param[
                           't_ref_cool'],
                        self.param['t_end'] / self.param['t_ref_cool']],
                       [0.5, 1.0],
                       [self.param["h_min"] / self.param["h_ref"],
                        self.param["h_max"] / self.param["h_ref"]]]

        self.num_elem = [1000,  # number of elements in x
                         1000,  # number of elements in y
                         100,  # number of elements in z
                         1000,  # number of elements in t
                         100,  # number of elements in eta
                         100]  # number of elements in h

        self.vs = create_meshes(self.num_elem, self.ords, self.ranges, vs_heating)#, refinement_t="cos")

    def __call__(self, pos_fixed=None, t_fixed=None, eta_fixed=None, h_fixed=None):
        # create FD matrices from meshes
        # t case
        t_dofs = np.array(self.vs[3].tabulate_dof_coordinates()[:].flatten())
        t_sort = np.argsort(t_dofs)
        M_t, _, D1_up_t = FD_matrices(t_dofs[t_sort])
        self.param['M_t'], self.param['D1_up_t'] = M_t[t_sort, :][:, t_sort], D1_up_t[t_sort, :][:, t_sort]
        self.param['bc_idx'] = np.where(t_dofs == self.ranges[3][0])[0]
        # eta case
        eta_dofs = np.array(self.vs[4].tabulate_dof_coordinates()[:].flatten())
        eta_sort = np.argsort(eta_dofs)
        M_eta, _, _ = FD_matrices(eta_dofs[eta_sort])
        self.param['M_eta'] = M_eta[eta_sort, :][:, eta_sort]
        # h case
        h_dofs = np.array(self.vs[5].tabulate_dof_coordinates()[:].flatten())
        h_sort = np.argsort(h_dofs)
        M_h, _, _ = FD_matrices(h_dofs[h_sort])
        self.param['M_h'] = M_h[h_sort, :][:, h_sort]

        # define inhomogeneous dirichlet IC in x, y, z, t, eta and h
        old_modes = remap_modes(pgd_heating=self.pgd_solution, vs=self.vs, param=self.param)
        self.param.update({'IC_x_old': old_modes[0]})
        self.param.update({'IC_y_old': old_modes[1]})
        self.param.update({'IC_z_old': old_modes[2]})
        self.param.update({'IC_t_old': old_modes[3]})
        self.param.update({'IC_eta_old': old_modes[4]})
        self.param.update({'IC_h_old': old_modes[5]})

        self.param.update(
            {'h': dolfin.interpolate(dolfin.Expression('x[0] * href', href=self.param['h_ref'], degree=4), self.vs[5])})

        solve_modes = ["FEM", "FEM", "FEM", "FD", "FD", "FD"]

        pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-SYZREtaH', name_coord=['X', 'Y', 'Z', 'T', 'Eta', 'H'],
                               modes_info=['T', 'Node', 'Scalar'],
                               Vs=self.vs, dom_fct=create_dom, bc_fct=create_bc, load=[],
                               param=self.param, rhs_fct=problem_assemble_rhs,
                               lhs_fct=problem_assemble_lhs,
                               probs=['x', 'y', 'z', 't', 'eta', 'h'], seq_fp=np.arange(len(self.vs)),
                               PGD_nmax=50, PGD_tol=1e-5)

        pgd_prob.MM = [0, 0, 0, self.param['M_t'], self.param['M_eta'], self.param['M_h']]  # for norms!

        pgd_prob.stop_fp = 'norm'
        pgd_prob.max_fp_it = 50
        pgd_prob.tol_fp_it = 1e-5
        pgd_prob.norm_modes = 'stiff'

        pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes)
        # pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes, settings = {"preconditioner": "amg", "linear_solver": "gmres"})

        print(pgd_prob.simulation_info)
        print('PGD Amplitude', pgd_prob.amplitude)

        pgd_s = pgd_prob.return_PGD()  # as PGD class instance

        cool_sol = pgd_s.evaluate(3, [0, 1, 2, 4, 5], [pos_fixed[0], pos_fixed[1], pos_fixed[2], eta_fixed,
                                                       h_fixed], 0).compute_vertex_values()[:]

        # add initial condition to heating phase
        for m_idx in range(len(self.param['IC_x_old'])):
            cool_sol += self.param['IC_x_old'][m_idx](pos_fixed[0]) * self.param['IC_y_old'][m_idx](pos_fixed[1]) * \
                        self.param[
                            'IC_z_old'][m_idx](pos_fixed[2]) \
                        * self.param['IC_t_old'][m_idx].compute_vertex_values()[:] * self.param["IC_eta_old"][m_idx](
                eta_fixed) * \
                        self.param["IC_h_old"][m_idx](h_fixed)

        return cool_sol, self.param, self.vs
