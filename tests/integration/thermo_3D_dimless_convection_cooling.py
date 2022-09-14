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

    def __init__(self, pgd_heating=None, param={}):
        self.pgd_solution = pgd_heating
        self.param = param

        self.ord = 1  # order for each mesh
        self.ords = [self.ord, self.ord, self.ord, self.ord, self.ord, self.ord]

        self.ranges = [[0., self.param["L"] / self.param['x_ref']],
                       [0., self.param["W"] / self.param['y_ref']],
                       [0., self.param["H"] / self.param['z_ref']],
                       [(self.pgd_solution.mesh[3].dataX * param['r_ref'] / param['vel'])[-1] / self.param['t_ref'],
                        self.param['t_end'] / self.param['t_ref']]
                       [0.5, 1.],
                       [self.param["h_min"] / self.param["h_ref"],
                        self.param["h_max"] / self.param["h_ref"]]]

        self.num_elem = [1000,  # number of elements in x
                         1000,  # number of elements in y
                         100,  # number of elements in z
                         1000,  # number of elements in t
                         100,  # number of elements in eta
                         100]  # number of elements in h

        self.meshes, self.vs = self.create_meshes(self.num_elem, self.ords, self.ranges)

    def __call__(self, pos_fixed=None, t_fixed=None, eta_fixed=None, h_fixed=None):
        # create FD matrices from meshes
        # t case
        t_dofs = np.array(self.vs[3].tabulate_dof_coordinates()[:].flatten()) / self.param['t_ref']
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

        solve_modes = ["FEM", "FEM", "FEM", "FD", "FD", "FD"]

        pgd_prob = PGDProblem1(name='1DHeatEqu-PGD-SYZREtaH', name_coord=['S', 'Y', 'Z', 'R', 'Eta', 'H'],
                               modes_info=['T', 'Node', 'Scalar'],
                               Vs=self.vs, dom_fct=create_dom, bc_fct=create_bc, load=[q_s, q_y, q_z, q_r, q_eta, q_h],
                               param=self.param, rhs_fct=problem_assemble_rhs, lhs_fct=problem_assemble_lhs,
                               probs=['s', 'y', 'z', 'r', 'eta', 'h'], seq_fp=np.arange(len(self.vs)),
                               PGD_nmax=20, PGD_tol=1e-5)

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
        return
