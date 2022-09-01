import logging

import numpy as np

import dolfin

from scipy.sparse import spdiags

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
        self.dims = dims  # list of dims

        # computed by class functions
        self.PGD_func = []  # solution of solve_PGD
        self.alpha = []  # joined norms of mode sets part of solution
        self.amplitude = []  # amplitude of PGD problem

        self.PGD_modes = None

        self.max_fp_it = 50  # maximum number of fixed point iteration steps
        self.tol_fp_it = 1e-5  # tolerance of fixed point iteration
        self.tol_abs = 1e-6 # absolute tolerance of fixed point iteration (norm criterion)
        self.stop_fp = 'norm' # FP break criterion "norm" or "delta"
        self.fp_init = '' # mode initialization " " (=Ones) or "randomized" (=RAND)
        self.norm_modes = 'l2' # norming of modes "no", "l2" or "stiff"

        self.simulation_info = 'PGD solver option: PGD_nmax %s / PGD tolerance %s and max FP iterations %s and FP tolerance %s; \n' % (
        self.PGD_nmax, self.PGD_tol, self.max_fp_it, self.tol_fp_it)

        self.solve_mode =	{ # dictionary to decide which solver is used
            "FEM": "FEM",
            "direct": "direct",
            "FD": "FD"
        }

        self.MM = [] # Mass matrix for norms if FD is used!
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

    def get_Fsinit(self, V, bc=None, solve_modes=None):
        '''
            create initialized functions with one including boundary conditions if there exist
            :param V: list of function spaces or only one fct space for which init fct should be computed
            :param bc: list of corresponding fenics boundary conditions
            :param solve_modes: if None or FEM dolfin.norm if FD use M matrices for norm has to be given in problem!!
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

                if dimension_dim == 1: # scalar Function Space
                    Fs_init[dim] = dolfin.interpolate(dolfin.Expression('1.0', degree=0), V=V[dim])

                    if bc[dim] != 0: #apply boundary conditions
                        if isinstance(bc[dim], list):
                            for i in range(len(bc[dim])):
                                bc[dim][i].apply(Fs_init[dim].vector())
                        else:
                            bc[dim].apply(Fs_init[dim].vector())

                    if self.fp_init.lower()=='randomized': # as in Chadys matlab code
                        idx = np.where(Fs_init[dim].vector()[:]!=0)[0] # idx without boundary
                        Fs_init[dim].vector()[idx] = np.random.rand(len(idx))

                    # normalize
                    if solve_modes != None and solve_modes[dim] == "FD":
                        norm_Fs_init = np.sqrt(Fs_init[dim].vector()[:].transpose() @ self.MM[dim] @ Fs_init[dim].vector()[:]) # norm with given M matrix
                    else:
                        norm_Fs_init = dolfin.norm(Fs_init[dim]) # classical l2 norm
                    Fs_init[dim].vector()[:] *= 1./norm_Fs_init # normalization
                    self.logger.debug('Fs_init[dim]: %s ', Fs_init[dim].compute_vertex_values()[:])

                elif dimension_dim == 2:  # VectorFunctionSpace!!!
                    Fs_init[dim] = dolfin.interpolate(dolfin.Expression(('1.0', '1.0'), element=V[dim].ufl_element()),
                                                  V=V[dim])  # , solver_type='mumps')
                    if bc[dim] != 0: #apply boundary conditions
                        if isinstance(bc[dim], list):
                            for i in range(len(bc[dim])):
                                bc[dim][i].apply(Fs_init[dim].vector())
                        else:
                            bc[dim].apply(Fs_init[dim].vector())

                    if self.fp_init.lower()=='randomized':
                        idx = np.where(Fs_init[dim].vector()[:] != 0)[0]  # idx without boundary
                        Fs_init[dim].vector()[idx] = np.random.rand(len(idx))

                    Fs_init[dim].vector()[:] *= 1. / dolfin.norm(Fs_init[dim])  # normalization with l2 norm (no FD option here just for 1D)
                elif dimension_dim == 3:  # VectorFunctionSpace!!!
                    Fs_init[dim] = dolfin.interpolate(
                        dolfin.Expression(('1.0', '1.0', '1.0'), element=V[dim].ufl_element()),
                        V=V[dim])  # , solver_type='mumps')

                    if bc[dim] != 0:  # apply boundary conditions
                        if isinstance(bc[dim], list):
                            for i in range(len(bc[dim])):
                                bc[dim][i].apply(Fs_init[dim].vector())
                        else:
                            bc[dim].apply(Fs_init[dim].vector())

                    if self.fp_init.lower()=='randomized':
                        idx = np.where(Fs_init[dim].vector()[:] != 0)[0]  # idx without boundary
                        Fs_init[dim].vector()[idx] = np.random.rand(len(idx))

                    Fs_init[dim].vector()[:] *= 1. / dolfin.norm(Fs_init[dim])  # normalization with l2 norm (no FD options here just in 1D)
                else:
                    self.logger.error('ERROR DIMENSION NOT defined!!!!!!!!!!!')
                    raise ValueError('ERROR DIMENSION NOT defined!!!!!!!!!!!')

            elif check_string.split(' ')[0] == '<tensor':
                self.logger.error('ERROR TENSOR function spaces not defined!!!!!')
                raise ValueError('ERROR TENSOR function spaces not defined!!!!!')
            else:
                # scalar function space for each dimension the same
                Fs_init[dim] = dolfin.interpolate(dolfin.Expression('1.0', degree=0), V=V[dim])

                if bc[dim] != 0:
                    if isinstance(bc[dim],list):
                        for i in range(len(bc[dim])):
                            bc[dim][i].apply(Fs_init[dim].vector())
                    else:
                        bc[dim].apply(Fs_init[dim].vector())

                if self.fp_init.lower()=='randomized': # as in Chadys matlab code
                    idx = np.where(Fs_init[dim].vector()[:] != 0)[0]  # idx without boundary
                    Fs_init[dim].vector()[idx] = np.random.rand(len(idx))

                if solve_modes != None and solve_modes[dim] == "FD":
                    norm_Fs_init = np.sqrt(Fs_init[dim].vector()[:].transpose() @ self.MM[dim] @ Fs_init[dim].vector()[:])  # norm with given M matrix
                else:
                    norm_Fs_init = dolfin.norm(Fs_init[dim])  # classical l2 norm

                self.logger.debug('Fs_init[dim] before normalization: %s', Fs_init[dim].compute_vertex_values()[:])
                Fs_init[dim].vector()[:] *= 1. / norm_Fs_init  # normalization with l2 norm
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

            # initialize Functions including boundary conditions!
            self.logger.info("enrichment step %s ", n_enr)
            Fs_init = self.get_Fsinit(self.V, self.bc, solve_modes)

            norm_Fs = np.ones(self.num_pgd_var)
            for i in range(self.num_pgd_var):
                norm_Fs[i] = dolfin.norm(Fs_init[i]) # TODO: check if repeated!
            delta = np.ones(self.num_pgd_var)

            # check residuum error computed with Fs_init
            res=[]
            for dim in range(self.num_pgd_var):
                if solve_modes is None or solve_modes[dim]==self.solve_mode["FEM"]:
                    var_F = dolfin.TestFunction(self.V[dim])
                    l = self.rhs_fct(Fs_init, var_F, Fs_init, self.meshes, self.dom, self.param, self.load,
                                     self.PGD_func,
                                     self.prob[dim], n_enr, dim)
                    if self.bc[dim] == 0:
                        ll = dolfin.assemble(l)[:]
                    else:
                        ll = dolfin.assemble(l)
                        if isinstance(self.bc[dim], list):
                            for i in range(len(self.bc[dim])):
                                self.bc[dim][i].apply(ll)
                        else:
                            self.bc[dim].apply(ll)
                        ll = ll[:]
                else:
                    ll = self.rhs_fct(Fs_init, Fs_init, Fs_init, self.meshes, self.dom, self.param, self.load, self.PGD_func,
                             self.prob[dim], n_enr, dim) # gives directly vector
                res.append(ll.transpose() @ ll)
            res_error = np.sqrt(np.sum(res))
            if res_error < 1e-10:
                msg = "Residuum error %s smaller 1e-10 in enrichment step number %s\n STOPP"
                self.logger.info(msg % (res_error, n_enr))
                self.simulation_info += f'<<<before enrichment step {n_enr} residuum norm smaller 1e-10: {res_error} STOP >>>\n'
                break

            # FP iteration
            Fs, norm_Fs = self.FP_solve(Fs_init, norm_Fs, delta, n_enr, _problem, solve_modes, settings)

            self.logger.debug('update PGD_func: old lenght: %s', len(self.PGD_func[0]))

            # normalization and adding new modes different methods possible
            normU=np.prod(norm_Fs)
            if self.norm_modes.lower()=='no':
                # no normalization at all
                for dim in range(self.num_pgd_var):
                    self.PGD_func[dim].append(Fs[dim])
                self.alpha.append(1.0) # no normalization

            elif self.norm_modes.lower()=='stiff':
                # norming of modes with stiffness

                Fs_normalized = np.copy(Fs) # normalized via l2 norm
                for dim in range(self.num_pgd_var):
                    Fs_normalized[dim].vector()[:] *= 1/norm_Fs[dim]

                # second normalization (assemble left hand side for last problem with new modes)
                fct_F = Fs_normalized[-1]
                var_F = Fs_normalized[-1]
                # define DGL
                a = self.lhs_fct(fct_F, var_F, Fs_normalized, self.meshes, self.dom, self.param, self.prob[-1],
                                 self.num_pgd_var)
                if solve_modes != None and solve_modes[-1] == self.solve_mode["FD"]:
                     norm_aux = var_F.vector()[:].transpose() @ a @ fct_F.vector()[:] # a== matrix
                elif solve_modes != None and solve_modes[-1] == self.solve_mode["direct"]:
                     norm_aux = a # scalar
                else:
                    norm_aux = dolfin.assemble(a) # a== form
                norm_fac = np.sqrt(np.absolute(norm_aux))**(1./self.num_pgd_var)
                self.alpha.append(np.prod(norm_Fs) * norm_fac ** self.num_pgd_var)  # joined norm factor

                for dim in range(self.num_pgd_var):
                    Fs_normalized[dim].vector()[:] *= 1. / norm_fac
                    Fs_normalized[dim].vector()[:] *= self.alpha[-1] ** (1. / self.num_pgd_var) # multiply each mode by a part of the joined norm
                    self.PGD_func[dim].append(Fs_normalized[dim])

                ## comment to normalization: alpha * F1/(|F1|*norm_fac) * F2/(|F2|*norm_fac) ... == [alpha^(1/dim) * F1/(|F1|*norm_fac)] * [alpha^(1/dim) * F2/(|F2|*norm_fac)] ... = R1neu * R2neu

            elif self.norm_modes.lower()=='l2':
                # norming of modes with l2 norm
                self.alpha.append(normU) # joined norm factor

                norm_all = np.prod(norm_Fs) ** (1. / self.num_pgd_var)
                for dim in range(self.num_pgd_var):
                    fac_Fs_i = norm_all / norm_Fs[dim]
                    # update PGD functions
                    tmp = dolfin.Function(self.V[dim])
                    tmp.vector().axpy(fac_Fs_i, Fs[dim].vector())
                    # print("newPGD"+str(dim)+" ", tmp.compute_vertex_values())
                    self.PGD_func[dim].append(tmp)
                ## comment to normalization: [norm_all * F1/|F1|] * [norm_all * F2/|F2|] ... = R1neu * R2neu with norm_all^dim = prod|F_i|!!!


            self.logger.debug('update PGD_func: new lenght: %s', len(self.PGD_func[0]))
            # input('end enrichment step')

            # compute norms and convergence criteria
            normConv.append(normU)
            relConv.append(normU / normConv[0])  # relative criterium start with 1.0

            self.logger.info("PGD modes updated: normU=%s; relNorm=%s; tol=%s; res_error=%s", normU, relConv[n_enr], self.PGD_tol, res_error)
            # input('end enrichment loop')
            # check convergence prod_i(||F^(n_enr)_i||) (similar to an displacement criterium)
            # if (normU < 1e-26):
            if relConv[n_enr] < self.PGD_tol:
                msg = "Convergence reached (normU = %s relative %s [res_error %s]), enriched basis number %s"
                self.logger.info(msg % (normU, relConv[n_enr], res_error, n_enr))
                self.logger.info("Convergence norms: %s; %s" % (normConv, relConv))
                break
            elif n_enr == self.PGD_nmax:
                self.logger.error(
                    "ERROR: Convergence not reached (normU = %s [res_error %s]) BUT PGD_nmax reached --> increase PGD_nmax",
                    normU, res_error)
                raise ValueError("Convergence not reached BUT Nmax reached!!")

        # save result in class instance
        self.amplitude = relConv
        self.PGD_modes = len(self.PGD_func[0])

        return self

    def FP_solve(self, Fs_init, norm_Fs, delta, n_enr, _problem, solve_modes, settings):
        '''
            compute Fixed Point iteration for all dims in sequence fp_seq
        :param Fs_init: initialized functions
        :param norm_Fs: norms of Fs
        :param delta: list dimension PGD_dim
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
                            prm["newton_solver"]["linear_solver"] = "mumps"  # "direct" #"gmres"
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
                            prm["linear_solver"] = "mumps"  # "direct" #"gmres"
                            for key, value in settings.items():
                                prm[key] = value
                            solver.solve()
                    elif solve_modes[dim] == self.solve_mode["direct"]:
                        fct_F = self.direct_solve(a, l, dim)
                    elif solve_modes[dim] == self.solve_mode["FD"]:
                        fct_F = self.FD_solve(a, l, dim)
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
                    elif solve_modes[dim] == self.solve_mode["FD"]:
                        fct_F = self.FD_solve(a, l, dim)
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
                if solve_modes != None and solve_modes[dim] == self.solve_mode["FD"]:
                        norm_Fs[dim] = np.sqrt(Fs[dim].vector()[:].transpose() @ self.MM[dim] @ Fs[dim].vector()[:]) # use FD mass matrix
                else:
                    norm_Fs[dim] = dolfin.norm(Fs[dim])  # classical FEM L2 norm

                self.logger.debug("norm for %s : %s", self.prob[dim], norm_Fs[dim])

            self.logger.debug("convergence type for fixed point iteration: %s (changeable in self.stop_fp)",
                             self.stop_fp)
            # checking FP iteration (delta or norm)
            if self.stop_fp.lower() == 'delta':
                for dim in range(self.num_pgd_var):
                    # delta between old solution
                    delta_tmp = np.absolute(Fs[dim].vector()[:] - Fs_init[dim].vector()[:])
                    max_index = np.argmax(delta_tmp)
                    if np.absolute(Fs[dim].vector()[max_index]) < 1e-8:
                        delta[dim] = delta_tmp.max()
                    else:
                        delta[dim] = delta_tmp.max() / np.absolute(Fs[dim].vector()[max_index])  # relative
                self.logger.debug(f"absolute delta {delta_tmp}, relative delta {delta}")

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
                # new set of solutions Fs old set of solutions Fs_init compute error after matlab code ghnatios (calc_diff_R(R,Rold))
                newnew, newold, oldold = 1, 1, 1

                for d in range(self.num_pgd_var):
                    if solve_modes != None and solve_modes[d] == 'FD':
                        # use given mass matrices for norms
                        newnew *= Fs[d].vector()[:].transpose() @ self.MM[d] @ Fs[d].vector()[:]
                        newold *= Fs[d].vector()[:].transpose() @ self.MM[d] @ Fs_init[d].vector()[:]
                        oldold *= Fs_init[d].vector()[:].transpose() @ self.MM[d] @ Fs_init[d].vector()[:]
                    else:
                        # use fencis assemble/norm
                        newnew *= dolfin.norm(Fs[d])**2 # same as dolfin.assemble(Fs[d]*Fs[d]* dolfin.dx(self.meshes[d]))
                        newold *= dolfin.assemble(dolfin.inner(Fs[d],Fs_init[d]) * dolfin.dx(self.meshes[d]))
                        oldold *= dolfin.norm(Fs_init[d])**2

                max_error = np.sqrt(np.absolute(newnew+oldold-2*newold))

                if max_error < self.tol_fp_it :
                    self.logger.info(
                        f"fix point iteration converged !!! in number of steps: {fpi + 1} (error {max_error:8.6e})")
                    self.simulation_info += f'enrichment step {n_enr} fixed point iteration converged in {fpi + 1} / error: {max_error:8.6e} \n'
                    break
                elif (fpi < self.max_fp_it - 1):
                    self.logger.debug("fix point iteration not converged %s (max %s) (error %s)",
                                      fpi, self.max_fp_it, max_error)
                    Fs_init = np.copy(Fs)

                elif (fpi == self.max_fp_it - 1):
                    self.logger.error(
                        f"ERROR: fix point iteration in maximum number of iterations NOT converged (enrichment loop {n_enr}) (error {max_error:8.6e})")
                    self.simulation_info += f'<<<enrichment step {n_enr} fixed point iteration NOT converged in {fpi + 1} / error: {max_error:8.6e} >>>\n'
                    # input('press enter to continue')
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

        # remark: alpha norming factor already partly multiplied in saved modes
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

    def FD_solve(self, A, B, dim):
        '''
            compute the FD solution of the problem Ax=B
        :param A: matrix left hand side matrix including boundary conditions!
        :param B: vector right hand side vector including boundary condition!
        :param dim: index of the corresponding function space
        :return: fct_F: new PGD function
        '''
        # Define the new function
        fct_F = dolfin.Function(self.V[dim])

        # Solve the equation
        vec = np.linalg.solve(A,B)

        # Set the DOFs to the solution
        fct_F.vector()[:] = vec
        return fct_F

# helper function for FD variant
def FD_matrices(x):
    N = len(x)
    e = np.ones(N)

    M = spdiags(e, 0, N, N).toarray()
    D2 = spdiags([e, e, e], [-1, 0, 1], N, N).toarray()
    D1_up = spdiags([e, e], [-1, 0], N, N).toarray()

    i = 0
    hp = x[i + 1] - x[i]

    M[i, i] = hp / 2

    D2[i, i] = -1 / hp
    D2[i, i + 1] = 1 / hp

    D1_up[i, i] = -1 / 2
    D1_up[i, i + 1] = 1 / 2

    for i in range(1, N - 1, 1):
        hp = x[i + 1] - x[i]
        hm = x[i] - x[i - 1]

        M[i, i] = (hp + hm) / 2

        D2[i, i] = -(hp + hm) / (hp * hm)
        D2[i, i + 1] = 1 / hp
        D2[i, i - 1] = 1 / hm

        D1_up[i, i] = (hp + hm) / (2 * hm)
        D1_up[i, i - 1] = -(hp + hm) / (2 * hm)

    i = N - 1
    hm = x[i] - x[i - 1]

    M[i, i] = hm / 2

    D2[i, i] = -1 / hm
    D2[i, i - 1] = 1 / hm

    D1_up[i, i] = (hp + hm) / (2 * hm)
    D1_up[i, i - 1] = -(hp + hm) / (2 * hm)
    return M, D2, D1_up
