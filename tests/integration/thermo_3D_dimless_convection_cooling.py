import dolfin
import numpy as np

from pgdrome.solver import PGDProblem1, FD_matrices


class PgdCooling:

    def create_meshes(num_elem, ord, ranges):
        '''
        :param num_elem: list for each PG CO
        :param ord: list for each PG CO
        :param ranges: list for each PG CO
        :return: meshes and V
        '''

        print('create meshes PGD cooling')

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

        # create boundarydomains
        boundarydomain_x = dolfin.MeshFunction("size_t", Vs[0].mesh(), Vs[0].mesh().topology().dim()-1)
        boundarydomain_x.set_all(0)
        boundarydomain_y = dolfin.MeshFunction("size_t", Vs[1].mesh(), Vs[1].mesh().topology().dim()-1)
        boundarydomain_y.set_all(0)
        boundarydomain_z = dolfin.MeshFunction("size_t", Vs[2].mesh(), Vs[2].mesh().topology().dim()-1)
        boundarydomain_z.set_all(0)

        class left_right(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return np.isclose(x,np.min(Vs[0].mesh().coordinates()[:])) or np.isclose(x,np.max(Vs[0].mesh().coordinates()[:]))
        
        class front_back(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return np.isclose(x,np.min(Vs[1].mesh().coordinates()[:])) or np.isclose(x,np.max(Vs[1].mesh().coordinates()[:]))
            
        class top_bottom(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return np.isclose(x,np.max(Vs[2].mesh().coordinates()[:])) or np.isclose(x,np.min(Vs[2].mesh().coordinates()[:]))

        Left_Right_dom = left_right()
        Left_Right_dom.mark(boundarydomain_x,1)  
        Front_Back_dom = front_back()
        Front_Back_dom.mark(boundarydomain_y,1)   
        Top_Bottom_dom = top_bottom()
        Top_Bottom_dom.mark(boundarydomain_z,1) 
       
        dom = [boundarydomain_x,boundarydomain_y,boundarydomain_z]

        return dom
    
    def create_bc(Vs,dom,param): 
           
       # Initial condition
       def init(x, on_boundary):
           return x < np.min(Vs[3].mesh().coordinates()[:]) + 1E-8

       initCond = dolfin.DirichletBC(Vs[3], dolfin.Constant(0.), init)

       return [0, 0, 0, initCond, 0, 0] # x,y,z,t,eta,h
    
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
        af = param['af']
        ar = param['af']
        h_g = af + ar
        h_1 = dolfin.Expression('x[0]-a_r \
                                ', degree=4, a_r=ar)
        h_2 = dolfin.Expression('L-a_f-x[0] \
                                ', degree=4, a_f=af, L=param['L'])
        
        r_last = (pgd_heating.mesh[3].dataX * param['r_ref'] / param['vel'])[-1]
            
        x_coords = vs[0].mesh().coordinates()
        
        # ==================================================================== #
        #                           COMPUTE MAPPING                            #
        # ==================================================================== #        
            
        x_funcs = list()
        y_funcs = list()
        z_funcs = list()
        t_funcs = list()  
        eta_funcs = list()
        h_funcs = list() 
            
        for m_idx in range(num_modes):
            x_funcs.append(dolfin.Function(vs[0]))
            y_funcs.append(dolfin.Function(vs[1]))
            z_funcs.append(dolfin.Function(vs[2]))
            eta_funcs.append(dolfin.Function(vs[4]))
            h_funcs.append(dolfin.Function(vs[5]))
            
            x_mode = np.zeros(len(x_coords))
            for i,x in enumerate(x_coords):
                u_x = 0
                
                # left side
                s_temp = x / h_1(r_last)
                if s_temp <= 1:
                    u_x = PGD_modes[0][m_idx](s_temp)
                # center
                else: 
                    s_temp = (x - h_1(r_last)) / h_g + 1
                    if s_temp > 1 and s_temp <= 2:
                        u_x = PGD_modes[0][m_idx](s_temp)
                    # right side
                    else:
                        s_temp = (x - h_1(r_last) - h_g) / h_2(r_last) + 2
                        u_x = PGD_modes[0][m_idx](s_temp-1E-8)
                        
                x_mode[i] = u_x
            
            
            x_funcs[m_idx].vector()[:] = x_mode[dolfin.dof_to_vertex_map(vs[0])]
            
            # for heating fixed
            y_funcs[m_idx] = dolfin.interpolate(PGD_modes[1][m_idx], vs[1])
            z_funcs[m_idx] = dolfin.interpolate(PGD_modes[2][m_idx], vs[2])
            eta_funcs[m_idx] = dolfin.interpolate(PGD_modes[4][m_idx], vs[4])
            h_funcs[m_idx] = dolfin.interpolate(PGD_modes[5][m_idx], vs[5])
            
            t_funcs.append(dolfin.Function(vs[3]))
            
            print(len(vs[3].mesh().coordinates()))
            print(len(t_funcs[0].vector()))
            
            tmp = np.zeros(len(vs[3].mesh().coordinates()))
            tmp[-1] = PGD_modes[3][m_idx](r_last)
            
            t_funcs[m_idx].vector()[:] = tmp
            
            # t_funcs[m_idx].vector()[:] = np.zeros(len(t_funcs[m_idx].vector()))
            # t_funcs[m_idx].vector()[-1] = PGD_modes[3][m_idx](r_last)
            
            
            
            # y_funcs[m_idx].vector()[:] = PGD_modes[1][m_idx].compute_vertex_values()[dolfin.dof_to_vertex_map(vs[1])]
            # z_funcs[m_idx].vector()[:] = PGD_modes[2][m_idx].compute_vertex_values()[dolfin.dof_to_vertex_map(vs[2])]
            
            # t_funcs.append(dolfin.interpolate(dolfin.Expression('(x[0] <= turnOff + 1E-8) ? value : 0 \
            #             ', degree=4, turnOff=prms["ranges"]["t"]["min"], value=PGD_modes[3][m_idx](r_last)),vs[3]))
                        
            # eta_funcs[m_idx].vector()[:] = PGD_modes[4][m_idx].compute_vertex_values()[dolfin.dof_to_vertex_map(vs[4])]
            # h_funcs[m_idx].vector()[:] = PGD_modes[5][m_idx].compute_vertex_values()[dolfin.dof_to_vertex_map(vs[5])]
            
        
        
        # override old functions with functions regarding the actual vs    
        PGD_modes[0] = x_funcs
        PGD_modes[1] = y_funcs
        PGD_modes[2] = z_funcs
        PGD_modes[3] = t_funcs
        PGD_modes[4] = eta_funcs
        PGD_modes[5] = h_funcs
                        
        return PGD_modes
    
    def problem_assemble_lhs(fct_F,var_F,Fs,meshes,dom,param,typ,dim):

        # define measures
        ds_x = dolfin.Measure('ds', domain=meshes[0], subdomain_data=dom[0])
        ds_y = dolfin.Measure('ds', domain=meshes[1], subdomain_data=dom[1])
        ds_z = dolfin.Measure('ds', domain=meshes[2], subdomain_data=dom[2])

        alpha_r1 = Fs[3].vector()[:].transpose() @ param['M_r'] @ Fs[3].vector()[:]
        alpha_r2 = Fs[3].vector()[:].transpose() @ param['D1_up_r'] @ Fs[3].vector()[:]
        alpha_eta1 = Fs[4].vector()[:].transpose() @ param['M_eta'] @ Fs[4].vector()[:]
        alpha_h1 = Fs[5].vector()[:].transpose() @ param['M_h'] @ Fs[5].vector()[:]
        alpha_h2 = (Fs[5].vector()[:] * param['h'].vector()[:]).transpose() @ param['M_h'] @ Fs[5].vector()[:]
                
        if typ == 'x':
            a  =  dolfin.Constant(dolfin.assemble(Fs[1] * Fs[1] * dolfin.dx(meshes[1]))
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
            a  =  dolfin.Constant(dolfin.assemble(Fs[0] * Fs[0] * dx_s(i))
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
    
    
    
    
    
    
    
    
    
    def __init__(self, pgd_heating=None, param={}):
        self.pgd_solution = pgd_heating
        self.param = param

        self.ord = 1  # order for each mesh
        self.ords = [self.ord, self.ord, self.ord, self.ord, self.ord, self.ord]

        self.param.update({'t_ref_cool': self.param['t_end']})
        self.ranges = [[0., self.param["L"] / self.param['x_ref']],
                       [0., self.param["W"] / self.param['y_ref']],
                       [0., self.param["H"] / self.param['z_ref']],
                       [(self.pgd_solution.mesh[3].dataX * param['r_ref'] / param['vel'])[-1] / self.param['t_ref_cool'],
                        self.param['t_end'] / self.param['t_ref_cool']]
                       [0.5, 1.],
                       [self.param["h_min"] / self.param["h_ref"],
                        self.param["h_max"] / self.param["h_ref"]]]

        self.num_elem = [500,  # number of elements in x
                         500,  # number of elements in y
                         50,  # number of elements in z
                         500,  # number of elements in t
                         50,  # number of elements in eta
                         50]  # number of elements in h

        self.meshes, self.vs = self.create_meshes(self.num_elem, self.ords, self.ranges)

    def __call__(self, pos_fixed=None, t_fixed=None, eta_fixed=None, h_fixed=None):
        # create FD matrices from meshes
        # t case
        t_dofs = np.array(self.vs[3].tabulate_dof_coordinates()[:].flatten())
        t_sort = np.argsort(t_dofs)
        M_t, _, D1_up_t = FD_matrices(t_dofs[t_sort])
        self.param['M_t'], self.param['D1_up_t'] = M_t[t_sort, :][:, t_sort], D1_up_t[t_sort, :][:, t_sort]
        self.param['bc_idx'] = np.where(t_dofs == 0)[0]
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

        # define nonhomogeneous dirichlet IC in x, y, z, t, eta and h  
        old_modes = self.remap_modes(pgd_heating=self.pgd_solution, vs=self.vs, param=self.param)
        self.param.update({'IC_x': old_modes[0]})
        self.param.update({'IC_y': old_modes[1]})
        self.param.update({'IC_z': old_modes[2]})
        self.param.update({'IC_t': old_modes[3]})
        self.param.update({'IC_eta': old_modes[4]})
        self.param.update({'IC_h': old_modes[5]})

        solve_modes = ["FEM", "FEM", "FEM", "FD", "FD", "FD"]

        pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-SYZREtaH', name_coord=['S', 'Y', 'Z', 'R', 'Eta', 'H'],
                               modes_info=['T', 'Node', 'Scalar'],
                               Vs=self.vs, dom_fct=self.create_dom, bc_fct=self.create_bc, load=[],
                               param=self.param, rhs_fct=problem_assemble_rhs, lhs_fct=problem_assemble_lhs,
                               probs=['s', 'y', 'z', 'r', 'eta', 'h'], seq_fp=np.arange(len(self.vs)),
                               PGD_nmax=5, PGD_tol=1e-5)

        pgd_prob.MM = [0, 0, 0, self.param['M_r'], self.param['M_eta'], self.param['M_h']]  # for norms!

        pgd_prob.stop_fp = 'norm'
        pgd_prob.max_fp_it = 50
        pgd_prob.tol_fp_it = 1e-5
        pgd_prob.norm_modes = 'stiff'

        pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes)
        # pgd_prob.solve_PGD(_problem='linear', solve_modes=solve_modes, settings = {"preconditioner": "amg", "linear_solver": "gmres"})

        print(pgd_prob.simulation_info)
        print('PGD Amplitude', pgd_prob.amplitude)

        pgd_s = pgd_prob.return_PGD()  # as PGD class instance
        
        return pgd_s, self.param
