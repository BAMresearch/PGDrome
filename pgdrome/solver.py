import logging

import numpy as np

import dolfin

from pgdrome.model import PGD


class PGDProblem1:

    def __init__(self, name=None, name_coord=[], modes_info=[], Vs=[], dom_fct=None,
                 bc_fct=None, load=[], param=None, rhs_fct=None, lhs_fct=None,
                 probs=[], seq_fp=[], PGD_nmax=20, PGD_tol=1e-10,
                 num_elem=[], order=[], ranges=[], dims=[], *args, **kwargs):
        '''
            :param name: always needed
            :param name_coord: always needed
            :param modes_info: always needed
            :param Vs: always needed
            :param dom_fct: always needed
            :param bc_fct: fct concerning problem setting
            :param load: fct concerning problem setting
            :param param: fct concerning problem setting
            :param rhs_fct: fct concerning problem setting
            :param lhs_fct: fct concerning problem setting
            :param probs: solver parameter
            :param seq_fp: solver parameter
            :param PGD_nmax: solver parameter
            :param PGD_tol: solver parameter
            :param num_elem: if meshing in class is used
            :param order: if meshing in class is used
            :param ranges: if meshing in class is used
            :param dims: if meshing in class is used
            :param args:
            :param kwargs:
        '''
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.name = name

        self.name_coord = name_coord  # list of abbreviation of PGD CO names e.g ['X','E'] -> for pxdmf
        self.modes_info = modes_info  # list with infos for pxdmf file e.g. ['U_x', 'Node', 'Scalar']

        self.num_pgd_var = len(self.name_coord)  # number of PGD COs

        # IMPORTANT USE ALWAYS THE SAME ORDER!!! Vs,num_elem ... if one is not given put 0
        self.V = [0] * self.num_pgd_var
        self.meshes = [0] * self.num_pgd_var  # depending on Vs
        if Vs != []:
            self.V = Vs  # list of fencis Functionspaces (0 if not given)
        for idx, vv in enumerate(Vs):
            if vv == 0:
                self.meshes[idx] = vv  # list of fenics meshes (0 if not given)
            else:
                self.meshes[idx] = vv.mesh()
        self.dom_fct = dom_fct  # list of domains if there are some
        self.bc_fct = bc_fct  # function which gives list of boundaries (use as input Vs,dom)
        # self.load_fct = load_fct  # function which gives list of separated load (use as input Vs,param)
        # self.param_fct = param_fct  # function which gives list of parameter (use as input Vs)
        self.load = load  # list of expressions for separated loads
        self.param = param  # dictionary including all needed parameters for assemble (including expressions!!)
        self.rhs_fct = rhs_fct  # function which gives the right hand side (input=)
        self.lhs_fct = lhs_fct  # function which gives the left hand side (input=) and a flag if the PDE is in the strong form

        self.prob = probs  # list of problem strings e.g. ['r','s'] used in assemble fcts
        if len(seq_fp) == 0:
            self.seq_fp = list(range(0, self.num_pgd_var))  # default sequence!!
        else:
            self.seq_fp = seq_fp  # list of sequence acoording probs for fp e.g. [0,1]
        self.PGD_nmax = PGD_nmax  # max number of PGD enrichement steps (default = 20)
        self.PGD_tol = PGD_tol  # stopping criterion enrichment step (default = 1e-10)

        # or if meshes are created in instance: mesh parameters
        self.num_elem = num_elem  # list of number of elements per CO
        self.order = order  # list of order per CO
        self.ranges = ranges  # list of ranges per CO
        self.dims = dims  # list of dims (in the moment only 1D meshes included)

        # computed by class functions
        self.PGD_func = []  # solution of solve_PGD
        self.amplitude = []  # amplitude of PGD problem

        self.PGD_modes = None

        self.max_fp_it = 10  # maximum number of fixed point iteration steps
        self.tol_fp_it = 1e-3  # tolerance of fixed point iteration
        self.tol_abs = 1e-4 # absoloute tlerance of fixed point iteration (norm criterion)
        # self.stop_fp = 'norm' # convergence criterium of fixed point iteration possible
        # ('norm': |prod(norm(Fs_i^fpi)) - prod(norm(Fs_i^fpi-1)| relative and absolute)
        # 'delta': max_i(max(|Fs_i^fpi-Fs_i^fpi-1|)) absolute
        self.stop_fp = 'delta'

        self.simulation_info = 'PGD solver option: PGD_nmax %s / PGD tolerance %s and max FP iterations %s and FP tolerance %s; \n' % (
        self.PGD_nmax, self.PGD_tol, self.max_fp_it, self.tol_fp_it)

        self.solve_mode =	{ # dictionary to decide which solver is used
            "FEM": "FEM",
            "RK": "RK4",
            "RKmapping": "RK4mapping",
            "direct": "direct"
        }

    @property
    # @functools.lru_cache(maxsize=100, typed=True)
    def dom(self):
        ''' Computes list of bcs for solver using Vs '''
        self.logger.debug('in domain creation fct class property dom')
        if self.dom_fct:
            self.logger.debug('domain array: %s', self.dom_fct(self.V, self.param))
            return self.dom_fct(self.V, self.param)
        else:
            return 0  # no subdomains defined!!

    @property
    # @functools.lru_cache(maxsize=100, typed=True)
    def bc(self):
        ''' Computes list of bcs for solver using Vs '''
        self.logger.debug('in boundary creation fct class property bc')
        dom_tmp = self.dom
        self.logger.debug('boundary array: %s', self.bc_fct(self.V, dom_tmp, self.param))
        return self.bc_fct(self.V, dom_tmp, self.param)

    def get_Fsnull(self, V):
        '''
            create null functions
            :param V: list of function spaces or only one fct space for which null fct should be computed
            :return: Fs_null list of null modes corresponding to input list V
        '''
        self.logger.debug("in get_Fsnull")
        Fs_null = [[]] * len(V)
        # for dim in range(PGD_dim):
        #     Fs_null.append([])

        for dim in range(len(V)):
            # print('dim in Fsnull', dim)
            dimension_dim = V[dim].mesh().topology().dim()
            self.logger.debug('dimension of mesh: %s', dimension_dim)
            # separated Null functions for scalar FunctionSpace or VectorFunctionSpaces
            check_string = str(V[dim].ufl_function_space().ufl_element())
            self.logger.debug('Functionspace of dim %s is from type %s', dim, check_string)
            if check_string.split(' ')[0] == '<vector':
                if dimension_dim == 1:
                    Fs_null[dim] = dolfin.interpolate(dolfin.Expression('0.0', degree=1), V=V[dim])
                    # Fs_null[dim] = dolfin.project(dolfin.Expression('0.0', degree=1), V=V[dim], solver_type='mumps')
                elif dimension_dim == 2:  # VectorFunctionSpace!!!
                    Fs_null[dim] = dolfin.interpolate(dolfin.Expression(('0.0', '0.0'), element=V[dim].ufl_element()),
                                                      V=V[dim])
                    # Fs_null[dim] = dolfin.project(dolfin.Expression(('0.0', '0.0'), element=V[dim].ufl_element()), V=V[dim])
                elif dimension_dim == 3:  # VectorFunctionSpace!!!
                    Fs_null[dim] = dolfin.interpolate(
                        dolfin.Expression(('0.0', '0.0', '0.0'), element=V[dim].ufl_element()),
                        V=V[dim])
                    # Fs_null[dim] = dolfin.project(
                    #     dolfin.Expression(('0.0', '0.0', '0.0'), element=V[dim].ufl_element()),
                    #     V=V[dim], solver_type='mumps')
                else:
                    self.logger.error('ERROR DIMENSION for Vector function space NOT defined!!!!!!!!!!!')
                    raise ValueError('ERROR DIMENSION for Vector function space NOT defined!!!!!!!!!!!')

            elif check_string.split(' ')[0] == '<tensor':
                self.logger.error('ERROR TENSOR function spaces not defined!!!!!')
                raise ValueError('ERROR TENSOR function spaces not defined!!!!!')
            else:
                # scalar function space for each dimension the same
                Fs_null[dim] = dolfin.interpolate(dolfin.Expression('0.0', degree=1), V=V[dim])

        return Fs_null

    def get_Fsinit(self, V, bc=None):
        '''
            create initialized functions with one including boundary conditions if there exist
            :param V: list of function spaces or only one fct space for which init fct should be computed
            :param bc: list of corresponding fenics boundary conditions
            :return: Fs_init list of initialized (one) modes
        '''
        self.logger.debug("in get_Fsinit (input lenghts: %s, %s)", len(V), len(bc))
        Fs_init = [[]] * len(V)
        if not bc:
            bc = [0] * len(V)

        for dim in range(len(V)):

            dimension_dim = V[dim].mesh().topology().dim()
            self.logger.debug('dimension of mesh: %s', dimension_dim)
            check_string = str(V[dim].ufl_function_space().ufl_element())
            self.logger.debug('Functionspace of dim %s is from type %s', dim, check_string)
            if check_string.split(' ')[0] == '<vector':

                if dimension_dim == 1:
                    if bc[dim] != 0:
                        Fs_init[dim] = dolfin.project(dolfin.Expression('1.0', degree=0),
                                                      V=V[dim], bcs=bc[dim], solver_type='mumps')
                    else:
                        Fs_init[dim] = dolfin.project(dolfin.Expression('1.0', degree=0), V=V[dim], solver_type='mumps')
                    self.logger.debug('Fs_init[dim]: %s ', Fs_init[dim].compute_vertex_values()[:])
                elif dimension_dim == 2:  # VectorFunctionSpace!!!
                    if bc[dim] != 0:
                        Fs_init[dim] = dolfin.project(dolfin.Expression(('1.0', '1.0'), element=V[dim].ufl_element()),
                                                      V=V[dim], bcs=bc[dim], solver_type='mumps')
                    else:
                        Fs_init[dim] = dolfin.project(dolfin.Expression(('1.0', '1.0'), element=V[dim].ufl_element()),
                                                      V=V[dim], solver_type='mumps')
                elif dimension_dim == 3:  # VectorFunctionSpace!!!
                    if bc[dim] != 0:
                        Fs_init[dim] = dolfin.project(
                            dolfin.Expression(('1.0', '1.0', '1.0'), element=V[dim].ufl_element()),
                            V=V[dim], bcs=bc[dim], solver_type='mumps')
                    else:
                        Fs_init[dim] = dolfin.project(
                            dolfin.Expression(('1.0', '1.0', '1.0'), element=V[dim].ufl_element()),
                            V=V[dim], solver_type='mumps')
                else:
                    self.logger.error('ERROR DIMENSION NOT defined!!!!!!!!!!!')
                    raise ValueError('ERROR DIMENSION NOT defined!!!!!!!!!!!')

            elif check_string.split(' ')[0] == '<tensor':
                self.logger.error('ERROR TENSOR function spaces not defined!!!!!')
                raise ValueError('ERROR TENSOR function spaces not defined!!!!!')
            else:
                # scalar function space for each dimension the same
                if bc[dim] != 0:
                    Fs_init[dim] = dolfin.project(dolfin.Expression('1.0', degree=0),
                                                  V=V[dim], bcs=bc[dim], solver_type='mumps')
                else:
                    Fs_init[dim] = dolfin.project(dolfin.Expression('1.0', degree=0), V=V[dim], solver_type='mumps')
                self.logger.debug('Fs_init[dim]: %s ', Fs_init[dim].compute_vertex_values()[:])

        return Fs_init

    def solve_PGD(self, _problem='nonlinear', solve_modes=None, settings={"linear_solver":"mumps"}):
        '''
            create PGD solution enrichment loop calling fixed point iteration
            :param: _problem: select variationalSolver linear or nonlinear
            :param solve_modes: list of solvers to be used, if not given the standard solver is used
            :param settings: dict of the settings the solver should use
            :return: PGD solution as PGDClass instance plus save self.amplitude and self.PGD_func
        '''

        self.logger.debug('in solve_PGD')

        # some predefined functions (F_init nicht hier muss neu gemacht werden in jedem enrichment schritt!!)
        Fs_null = self.get_Fsnull(self.V)


        # enrichment loop
        # for n_enr in range(self.PGD_nmax):
        n_enr = -1
        while n_enr < self.PGD_nmax - 1:
            n_enr += 1
            if n_enr == 0:
                # initialize PGD solution
                self.PGD_func = list()
                for dim in range(self.num_pgd_var):
                    self.PGD_func.append(list())
                    # start PGD
                    normConv = list()
                    relConv = list()
                    normConv2 = list()
                    relConv2 = list()

            # initialize Functions including boundary conditions!
            self.logger.info("enrichment step %s ", n_enr)
            Fs_init = self.get_Fsinit(self.V, self.bc)

            norm_Fs = np.ones(self.num_pgd_var)
            for i in range(self.num_pgd_var):
                norm_Fs[i] = dolfin.norm(Fs_init[i])
            delta = np.ones(self.num_pgd_var)

            # FP iteration as extra function
            Fs, norm_Fs = self.FP_solve(Fs_init, norm_Fs, delta, Fs_null, n_enr, _problem, solve_modes, settings)

            self.logger.debug('update PGD_func: old lenght: %s', len(self.PGD_func[0]))
            # computed weighting factors
            normU = 1.0
            normPGD = 1.0
            norm_all = np.prod(norm_Fs) ** (1. / self.num_pgd_var)  # in paper version
            fac_Fs = np.zeros(self.num_pgd_var)
            for dim in range(self.num_pgd_var):
                if norm_Fs[dim] < 1e-8:
                    fac_Fs[dim] = 1.0
                else:
                    fac_Fs[dim] = norm_all / norm_Fs[dim]
                normU = normU * norm_Fs[dim]  # classical L2 norm without weighting of pgd modes

                # update PGD functions
                tmp = dolfin.Function(self.V[dim])
                tmp.vector().axpy(fac_Fs[dim], Fs[dim].vector())
                # print("newPGD"+str(dim)+" ", tmp.compute_vertex_values())
                self.PGD_func[dim].append(tmp)
                normPGD = normPGD * dolfin.norm(tmp)  # classical L2 norm with weighting of pgd modes

            self.logger.debug('update PGD_func: new lenght: %s', len(self.PGD_func[0]))


            # compute norms and convergence criteria
            normConv.append(normU)
            normConv2.append(normPGD)
            relConv.append(normU / normConv[0])  # relative criterium start with 1.0
            relConv2.append(normPGD / normConv2[0])

            self.logger.info("PGD modes updated: normU=%s; relNorm=%s; tol=%s", normU, relConv[n_enr], self.PGD_tol)
            self.logger.info("PGD modes: normPGD=%s relConv2=%s", normPGD, relConv2[n_enr])
            # input('end enrichment loop')
            # check convergence prod_i(||F^(n_enr)_i||) (similar to an displacement criterium)
            # if (normU < 1e-26):
            if relConv[n_enr] < self.PGD_tol:
                msg = "Convergence reached (normU = %s relative %s), enriched basis number %s"
                self.logger.info(msg % (normU, relConv[n_enr], n_enr))
                self.logger.info("Convergence norms: %s; %s" % (normConv, relConv))
                self.logger.info("Convergence norms 2: %s; %s" % (normConv2, relConv2))
                break
            elif n_enr == self.PGD_nmax:
                self.logger.error(
                    "ERROR: Convergence not reached (normU = %s) BUT PGD_nmax reached --> increase PGD_nmax",
                    normU)
                raise ValueError("Convergence not reached BUT Nmax reached!!")

        # save result in class instance
        self.amplitude = relConv
        self.PGD_modes = len(self.PGD_func[0])

        return self

    def FP_solve(self, Fs_init, norm_Fs, delta, Fs_null, n_enr, _problem, solve_modes, settings):
        '''
            compute Fixed Point iteration for all dims in sequence fp_seq
        :param Fs_init: initialized functions
        :param norm_Fs: norms of Fs
        :param delta: list dimension PGD_dim
        :param Fs_null: null functions
        :param n_enr: number of current enrichment step
        :param _problem: selects solver linear or nonlinear
        :param solve_modes: list of solvers to be used, if None the standard solver is used
        :param settings: dict of the settings the solver should use
        :return: Fs: list of new PGD functions
        :return: norm_Fs: updated list of norms of Fs
        '''
        self.logger.debug('in FP solve:')
        # ffc_options = {}
        ffc_options = {"optimize": True}  # parameters are set globally in parameters["form_compiler"]
        # local parameters are used as first and then added by additional global parameters

        Fs = np.copy(np.array(Fs_init, dtype=object))
        norm_Fs_init = np.copy(norm_Fs)

        # fixed point iteration
        for fpi in range(self.max_fp_it):

            self.logger.debug(f"##### Fixed_point iteration {fpi}:")
            for i in range(len(Fs_init)):
                self.logger.debug(f"Fs_{i}: {Fs_init[i].vector()[:]}")
            # input()

            for seq in range(len(self.seq_fp)):
                # for dim in range(PGD_dim): # changed to seq possible to change the sequence of the problems
                dim = self.seq_fp[seq]

                self.logger.debug("%s.problem for %s:", dim, self.prob[dim])
                fct_F = dolfin.Function(self.V[dim])
                var_F = dolfin.TestFunction(self.V[dim])

                if (len(np.where(norm_Fs < 1e-25)[0]) > 0):
                # if (len(np.where(norm_Fs < 1e-8)[0]) > 0):
                    self.logger.debug(
                        "one enrichment basis vanish --> DGL not possible to solve -> solution set to Zero! normFs=%s",
                        norm_Fs)
                    Fs[dim] = Fs_null[dim]
                    # print('Fs[dim]',len(Fs[dim].compute_vertex_values()[:]))
                    # compute norm and delta
                    delta[dim] = np.absolute(
                        Fs[dim].compute_vertex_values() - Fs_init[dim].compute_vertex_values()).max()
                    norm_Fs[dim] = dolfin.norm(Fs[dim])  # classical L2 norm

                else:
                    # define DGL
                    a = self.lhs_fct(fct_F, var_F, Fs, self.meshes, self.dom, self.param, self.prob[dim], dim)
                    l = self.rhs_fct(fct_F, var_F, Fs, self.meshes, self.dom, self.param, self.load, self.PGD_func,
                                     self.prob[dim], n_enr, dim)

                   # self.logger.debug('check RHS %s', dolfin.assemble(l)[:])

                    if self.bc[dim] == 0:
                        self.logger.debug('problem without boundary conditions')
                        if solve_modes is None or solve_modes[dim] == self.solve_mode["FEM"]: # standard FEM solver
                            if _problem.lower() == 'nonlinear':
                                # dolfin.solve(a - l == 0, Fs[dim])
                                J = dolfin.derivative(a - l, fct_F)
                                # # check condition number
                                # JJ = dolfin.assemble(J)
                                # print('condition number', np.linalg.cond(JJ.array()[:][:]))
                                problem = dolfin.NonlinearVariationalProblem(a - l, fct_F,
                                                                         J=J,
                                                                         form_compiler_parameters=ffc_options)
                                solver = dolfin.NonlinearVariationalSolver(problem)
                                prm = solver.parameters
                                for key, value in settings.items():
                                    prm["newton_solver"][key] = value                     
                                solver.solve()
                            elif _problem.lower() == 'linear':
                                # alternative use linear solver:
                                fct_F = dolfin.TrialFunction(self.V[dim])
                                a = self.lhs_fct(fct_F, var_F, Fs, self.meshes, self.dom, self.param, self.prob[dim], dim)
                                l = self.rhs_fct(fct_F, var_F, Fs, self.meshes, self.dom, self.param, self.load,
                                             self.PGD_func,
                                             self.prob[dim], n_enr, dim)
                                # AA = dolfin.assemble(a)
                                # # print('matrix',AA.array()[:][:])
                                # print('condition number', np.linalg.cond(AA.array()[:][:]))

                                fct_F = dolfin.Function(self.V[dim])
                                problem = dolfin.LinearVariationalProblem(a, l, fct_F, form_compiler_parameters=ffc_options)

                                solver = dolfin.LinearVariationalSolver(problem)
                                prm = solver.parameters
                                for key, value in settings.items():
                                    prm[key] = value
                                solver.solve()
                        elif solve_modes[dim] == self.solve_mode["direct"]:
                            fct_F = self.direct_solve(a, l, dim)
                        else:
                            self.logger.error("ERROR: solver %s doesn't exist", solve_modes[dim])
                    else:
                        self.logger.debug('problem with boundary conditions')
                        if solve_modes is None or solve_modes[dim] == self.solve_mode["FEM"]: # standard FEM solver
                            if _problem.lower() == 'nonlinear':
                                bc_tmp = self.bc[dim]
                                # print('check if we are here')
                                J = dolfin.derivative(a - l, fct_F)
                                # print('or here')
                                # # check condition number
                                # JJ = dolfin.assemble(J)
                                # print('JJ',JJ.array()[:][:])
                                # print('condition number', np.linalg.cond(JJ.array()[:][:]))
                                problem = dolfin.NonlinearVariationalProblem(a - l, fct_F, bcs=bc_tmp,
                                                                         J=J,
                                                                         form_compiler_parameters=ffc_options)
                                # problem = dolfin.NonlinearVariationalProblem(a - l, fct_F, bcs=bc_tmp, J=dolfin.derivative(a - l, fct_F),
                                #                                              form_compiler_parameters=ffc_options)

                                solver = dolfin.NonlinearVariationalSolver(problem)
                                prm = solver.parameters
                                for key, value in settings.items():
                                    prm["newton_solver"][key] = value
                                solver.solve()
                            elif _problem.lower() == 'linear':
                                # alternative use linear solver:
                                fct_F = dolfin.TrialFunction(self.V[dim])
                                a = self.lhs_fct(fct_F, var_F, Fs, self.meshes, self.dom, self.param, self.prob[dim], dim)
                                l = self.rhs_fct(fct_F, var_F, Fs, self.meshes, self.dom, self.param, self.load,
                                             self.PGD_func,
                                             self.prob[dim], n_enr, dim)
                                bc_tmp = self.bc[dim]

                                fct_F = dolfin.Function(self.V[dim])
                                problem = dolfin.LinearVariationalProblem(a, l, fct_F, bc_tmp,form_compiler_parameters=ffc_options)

                                solver = dolfin.LinearVariationalSolver(problem)
                                prm = solver.parameters
                                for key, value in settings.items():
                                    prm[key] = value
                                solver.solve()
                        elif solve_modes[dim] == self.solve_mode["direct"]:
                            fct_F = self.direct_solve(a, l, dim)
                        else:
                            self.logger.error("ERROR: solver %s doesn't exist", solve_modes[dim])

                    self.logger.debug("problem %s: F_%s_max: %s", self.prob[dim], str(dim),
                                      np.absolute(fct_F.vector()[:]).max())
                    self.logger.debug("problem %s: Finit_%s: %s =? %s", self.prob[dim], str(dim),
                                      Fs_init[dim].vector()[:], Fs[dim].vector()[:])
                    self.logger.debug("problem %s: Fs-Fs_init: %s", self.prob[dim],
                                      fct_F.vector()[:] - Fs_init[dim].vector()[:])

                    # compute norm and delta
                    Fs[dim] = fct_F
                    norm_Fs[dim] = dolfin.norm(Fs[dim])  # classical L2 norm

                    ### NEW correction
                    delta_tmp = np.absolute(Fs[dim].vector()[:] - Fs_init[dim].vector()[:])
                    max_index = np.argmax(delta_tmp)
                    if np.absolute(Fs[dim].vector()[max_index]) < 1e-3:
                        delta[dim] = delta_tmp.max()
                    else:
                        delta[dim] = delta_tmp.max() / np.absolute(Fs[dim].vector()[max_index])

                    # OLD used for paper
                    # norm_init = dolfin.norm(Fs_init[dim])
                    # if norm_Fs[dim] < 1e-8:
                    #     delta[dim] = np.absolute(Fs[dim].compute_vertex_values()[:] -
                    #                          Fs_init[dim].compute_vertex_values()[:]).max()
                    # else:
                    #     delta[dim] = np.absolute(1 / norm_Fs[dim] *
                    #                          Fs[dim].compute_vertex_values()[:] -
                    #                          1 / norm_init *
                    #                          Fs_init[dim].compute_vertex_values()[:]).max()

                self.logger.debug("norm for %s : %s", self.prob[dim], norm_Fs[dim])
                # print("F"+str(dim)+" "+self.prob[dim], Fs[dim].compute_vertex_values()[:])
                # print('Fs_init[dim]',Fs_init[dim].compute_vertex_values()[:])
                self.logger.debug("delta_%s: %s", dim, delta[dim])

            # norm_Fs_init = np.ones(self.num_pgd_var)
            # for i in range(len(norm_Fs_init)):
            #     norm_Fs_init[i] = dolfin.norm(Fs_init[i])
            self.logger.debug('norms Fs: %s, Fs_init %s, norm product Fs: %s, Fs_init: %s', norm_Fs, norm_Fs_init,
                              np.prod(norm_Fs), np.prod(norm_Fs_init))
            delta_norm = np.absolute(np.prod(norm_Fs) - np.prod(norm_Fs_init))
            norm_init = np.absolute(np.prod(norm_Fs_init))
            if np.prod(norm_Fs) > 1e-8:
                delta_norm_rel = delta_norm / norm_init
            else:
                delta_norm_rel = delta_norm

            # check convergence Fixed Point iteration
            self.logger.debug("Fixed point iteration %s: max.deltas=max(%s)=%s", fpi, delta, delta.max())
            self.logger.debug("Fixed point iteration %s: relative deltas in norms=%s (abs %s)", fpi, delta_norm_rel,
                             delta_norm)
            self.logger.debug("convergence type for fixed point iteration: %s (changeable in self.stop_fp)",
                             self.stop_fp)
            # input()

            if self.stop_fp.lower() == 'delta':
                if (len(np.where(delta > self.tol_fp_it)[0]) > 0 and fpi < self.max_fp_it - 1):
                    self.logger.debug("fix point iteration not converged %s (max %s)", fpi, self.max_fp_it)
                    Fs_init = np.copy(Fs)
                    norm_Fs_init = np.copy(norm_Fs)
                elif (len(np.where(delta > self.tol_fp_it)[0]) > 0 and fpi == self.max_fp_it - 1):
                    self.logger.error(
                        "ERROR: fix point iteration in maximum number of iterations NOT converged (enrichment loop %s)",
                        n_enr)
                    self.simulation_info += f'<<<enrichment step {n_enr} fixed point iteration NOT converged in {fpi + 1} / delta: {delta} >>>\n'
                    # input('press enter to continue')
                    break
                else:
                    self.logger.info("fix point iteration converged !!! in number of steps: %s (delta:%s)", fpi + 1,delta)
                    self.simulation_info += f'enrichment step {n_enr} fixed point iteration converged in {fpi + 1} / delta: {delta} \n'
                    break
            elif self.stop_fp.lower() == 'norm':
                # tol_abs = self.tol_fp_it

                if (delta_norm_rel < self.tol_fp_it or delta_norm < self.tol_abs):
                    self.logger.info("fix point iteration converged !!! in number of steps: %s (delta norm rel %s, abs: %s, init: %s)",
                                     fpi + 1, delta_norm_rel,delta_norm, norm_init)
                    self.simulation_info += f'enrichment step {n_enr} fixed point iteration converged in {fpi+1} / norms: {delta_norm_rel},{delta_norm} \n'
                    break
                elif (fpi < self.max_fp_it - 1):
                    self.logger.debug("fix point iteration not converged %s (max %s) (delta norm rel %s, abs: %s, init: %s)",
                                      fpi, self.max_fp_it, delta_norm_rel, delta_norm,norm_init)
                    Fs_init = np.copy(Fs)
                    norm_Fs_init = np.copy(norm_Fs)
                elif (fpi == self.max_fp_it - 1):
                    self.logger.error(
                        "ERROR: fix point iteration in maximum number of iterations NOT converged (enrichment loop %s) (delta norm rel %s, abs: %s, init: %s)",
                        n_enr, delta_norm_rel, delta_norm, norm_init)
                    self.simulation_info += f'<<<enrichment step {n_enr} fixed point iteration NOT converged in {fpi + 1} / norms: {delta_norm_rel},{delta_norm} >>>\n'
                    # input('press enter to continue')
                    break
                else:
                    self.logger.error( "ERROR:  something got wrong!!!")
                    break

            else:
                self.logger.error('stopping criterion not defined %s (self.stop_fp = "delta" or "norm")', self.stop_fp)
                raise ValueError('stopping criterion not defined %s (self.stop_fp = "delta" or "norm")')

        return Fs, norm_Fs

    def return_PGD(self):
        '''
            return solution as PGD CLASS
        :return:
        '''
        self.logger.info('in return PGD Model')
        num_elemt = np.zeros(len(self.meshes))
        for idx in range(len(self.meshes)):
            num_elemt[idx] = self.meshes[idx].num_cells()
        self.logger.info('Current mesh: %s ', num_elemt)

        solution = PGD(name=self.name, n_modes=self.PGD_modes, fmeshes=self.meshes, pgd_modes=self.PGD_func,
                            name_coord=self.name_coord, modes_info=self.modes_info, verbose=False )
        solution.problem = self
        solution.print_info()

        return solution

    def direct_solve(self, a, b, dim):
        '''
            compute the direct solution of the problem ax=b
        :param a: left hand side
        :param l: right hand side
        :param dim: index of the corresponding function space
        :return: fct_F: new PGD function
        '''
        # Define the new function
        fct_F = dolfin.Function(self.V[dim])

        # Solve the equation
        vec = b/a

        # Set the DOFs to the solution
        fct_F.vector()[:] = vec
        return fct_F
