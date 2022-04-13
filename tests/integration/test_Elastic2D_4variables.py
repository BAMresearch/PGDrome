# -*- coding: utf-8 -*-
"""
TEST: Elastic plate (2D) is modeled using PGD and FEM. The displacemetns 
computed by the two mehods are compared to check its working well.

"""

import unittest
import pytest
from dolfin import *
import numpy as np
import os
from pgdrome.solver import PGDProblem1


# MESHES
#==============================================================================
# Mesh: Geometry
#-----------------------------
# [[mesh_0, 'Family', degree, matID], [mesh_1, 'Family', degree, matID], ...]
def create_meshes(mesh_0,V_X,input_mesh):
    meshes = list()
    Vs = list()
    meshes.append(mesh_0)
    Vs.append(V_X)
    
    # Creating mesh for variables:
         
    for k in range(len(input_mesh)):
        if k == 0:
            print("Not written")
            
        elif len(input_mesh[k][0]) == 2:
            aux_mesh = IntervalMesh(input_mesh[k][1][0], input_mesh[k][0][0], input_mesh[k][0][1])
            aux_V = FunctionSpace(aux_mesh,input_mesh[k][2],input_mesh[k][3])
            meshes.append(aux_mesh) # Get out of the IF when k==0 is written
            Vs.append(aux_V) # Get out of the IF when k==0 is written
        
        else:
            print("Error: Not well defined")
            
    return meshes, Vs
        
# DIRICHLET BC:
#==============================================================================
def create_bc(Vs,dom,param):
    
    def BC_ConstDisp(x, on_boundary):
        tol = 1e-5
        return on_boundary and near(x[0], 0, tol) or near(x[1], 0, tol) or near(x[0], 1, tol)  
    
    # on_boundary: Dolfin flags nodes on the boundary with on_boundary
    u0 = Constant((0.,0.))
    bc = DirichletBC(Vs[0],u0,BC_ConstDisp)
    
    return [bc, 0, 0, 0]

# NEUMMAN BC: Domain definition
#==============================================================================
def create_dom(Vs, param):
    # Identify entities of diemension (dim-1) with a scalar(size_t) from the
    # mesh (Vs[0].mesh()):
    domi = MeshFunction("size_t", Vs[0].mesh(), Vs[0].mesh().topology().dim()-1)
    domi.set_all(0) # initialize the function to zero
    
    # Create classes for defining parts of the boundaries:
    class BC_Neumman(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-5
            return on_boundary and near(x[1], 1, tol) 

    # Initialize sub-domain instance:
    bnd_loadY = BC_Neumman()
    bnd_loadY.mark(domi, 1) # ID of the boundary
    
    return [domi, 0, 0, 0]

# STRAIN VECTOR
#==============================================================================
def epsilon(v):
    # Strain: epsilon_ij = 0.5*(du_i/dx_j + du_j/dx_i)
    strain_xx = grad(v)[0,0] #  = u[0].dx(0)
    strain_yy = grad(v)[1,1] #  = u[1].dx(1)
    strain_xy = 0.5* (grad(v)[0,1]+grad(v)[1,0]) #  = 0.5*(u[0].dx(1)+u[1].dx(0))
    
    return as_vector([strain_xx, strain_yy, 2*strain_xy])

# LEFT HAND SIDE:
#==============================================================================
def problem_assemble_lhs(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    
    # Computing R assuming S, T and V is known: 
    if typ == 'r':
        a = Constant(assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * param["nufunc"][0] * Fs[3] * dx(meshes[3]))) \
            * inner(param["aux_C"][0]*epsilon(fct_F),epsilon(var_F)) * dx(meshes[0]) \
            + Constant(assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * param["nufunc"][1] * Fs[3] * dx(meshes[3])) )\
            * inner(param["aux_C"][1]*epsilon(fct_F),epsilon(var_F)) * dx(meshes[0])
            
    # Computing S assuming R, T and V is known:
    if typ == 's':
        a = Constant(assemble(inner(param["aux_C"][0]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * param["nufunc"][0] * Fs[3] * dx(meshes[3])) )\
            * var_F* fct_F * dx(meshes[1]) \
            + Constant(assemble(inner(param["aux_C"][1]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * param["nufunc"][1] * Fs[3] * dx(meshes[3])) )\
            * var_F* fct_F * dx(meshes[1])
            
    # Computing T assuming R, S and V is known:
    if typ == 't':
        a = Constant(assemble(inner(param["aux_C"][0]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[3] * param["nufunc"][0] * Fs[3] * dx(meshes[3])) )\
            * var_F * param["Efunc"] * fct_F * dx(meshes[2]) \
            + Constant(assemble(inner(param["aux_C"][1]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[3] * param["nufunc"][1] * Fs[3] * dx(meshes[3])) )\
            * var_F * param["Efunc"] * fct_F * dx(meshes[2])
            
    # Computing V assuming R, S and T is known:
    if typ == 'v':
        a = Constant(assemble(inner(param["aux_C"][0]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) )\
            * var_F * param["nufunc"][0] * fct_F * dx(meshes[3]) \
            + Constant(assemble(inner(param["aux_C"][0]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) )\
            * var_F * param["nufunc"][1] * fct_F * dx(meshes[3])

    return a

# RIGHT HAND SIDE:
#==============================================================================
def problem_assemble_rhs(fct_F, var_F, Fs, meshes, dom, param, G, PGD_func, typ, nE, dim):
    
    ds = Measure('ds', domain=meshes[0], subdomain_data=dom[0], subdomain_id=1) # Integral over the boundary

    l = 0

    # Computing R assuming S, T and V is known: 
    if typ == 'r':
        # for ext in range(len(G[0][0])):
        l += Constant(assemble(param["Afunc"] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * dx(meshes[3])) )\
            * dot(G[0][0][0],var_F) * ds
                
        if nE > 0:
            for old in range(nE):
                l += -Constant(assemble(Fs[1] * PGD_func[1][old] * dx(meshes[1])) \
                    * assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])) \
                    * assemble(Fs[3] * param["nufunc"][0] * PGD_func[3][old] * dx(meshes[3])) )\
                    * inner(param["aux_C"][0]*epsilon(PGD_func[0][old]),epsilon(var_F)) * dx(meshes[0]) \
                    - Constant(assemble(Fs[1] * PGD_func[1][old] * dx(meshes[1])) \
                    * assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])) \
                    * assemble(Fs[3] * param["nufunc"][1] * PGD_func[3][old] * dx(meshes[3])) )\
                    * inner(param["aux_C"][1]*epsilon(PGD_func[0][old]),epsilon(var_F)) * dx(meshes[0])
                        
    # Computing S assuming R, T and V is known:
    if typ == 's':
        # for ext in range(len(G[0][1])):
        l += Constant(assemble(dot(G[0][0][0],Fs[0]) * ds) \
            * assemble(Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * dx(meshes[3])) )\
            * var_F * param["Afunc"] * dx(meshes[1])

        if nE > 0:
            for old in range(nE):
                l += -Constant(assemble(inner(param["aux_C"][0]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])) \
                    * assemble(Fs[3] * param["nufunc"][0] * PGD_func[3][old] * dx(meshes[3])) )\
                    * var_F * PGD_func[1][old] * dx(meshes[1]) \
                    - Constant(assemble(inner(param["aux_C"][1]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])) \
                    * assemble(Fs[3] * param["nufunc"][1] * PGD_func[3][old] * dx(meshes[3])) )\
                    * var_F * PGD_func[1][old] * dx(meshes[1])
                    
                   
    # Computing T assuming R, S and V is known:
    if typ == 't':
        # for ext in range(len(G[0][1])):
        l += Constant(assemble(dot(G[0][0][0],Fs[0]) * ds) \
            * assemble(param["Afunc"] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[3] * dx(meshes[3])) )\
            * var_F * dx(meshes[2])

        if nE > 0:
            for old in range(nE):
                l += -Constant(assemble(inner(param["aux_C"][0]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[1]*PGD_func[1][old] *dx(meshes[1])) \
                    * assemble(Fs[3] * param["nufunc"][0] * PGD_func[3][old] * dx(meshes[3])) )\
                    * var_F * param["Efunc"] * PGD_func[2][old] * dx(meshes[2]) \
                    - Constant(assemble(inner(param["aux_C"][1]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[1]*PGD_func[1][old] *dx(meshes[1])) \
                    * assemble(Fs[3] * param["nufunc"][1] * PGD_func[3][old] * dx(meshes[3])) )\
                    * var_F * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])

    # Computing V assuming R, S and T is known:
    if typ == 'v':
        # for ext in range(len(G[0][1])):
        l += Constant(assemble(dot(G[0][0][0],Fs[0]) * ds) \
            * assemble(param["Afunc"] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * dx(meshes[2])) )\
            * var_F * dx(meshes[3])

        if nE > 0:
            for old in range(nE):
                l += -Constant(assemble(inner(param["aux_C"][0]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[1]*PGD_func[1][old] *dx(meshes[1])) \
                    * assemble(Fs[2]*param["Efunc"]*PGD_func[2][old] *dx(meshes[2])) )\
                    * var_F * param["nufunc"][0] * PGD_func[3][old] * dx(meshes[3]) \
                    - Constant(assemble(inner(param["aux_C"][1]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[1]*PGD_func[1][old] *dx(meshes[1])) \
                    * assemble(Fs[2]*param["Efunc"]*PGD_func[2][old] *dx(meshes[2])) )\
                    * var_F * param["nufunc"][1] * PGD_func[3][old] * dx(meshes[3])
                            
    return l

# PGD PROBLEM:
#==============================================================================
def main(Vs):
    '''computation of PGD solution for given problem '''
    
    # PGD input
    #-----------------------------
    
    name_coord = ['X','A','E','P']

    prob = ['r','s','t','v'] # problems according problem_assemble_fcts
    seq_prob = [0, 1, 2, 3] # default sequence of Fixed Point iteration
    PGD_nmax = 15 # max number of PGD modes

    # DEFINE PARAMETERS:
    #-----------------------------
    C1 = as_matrix(np.array([[1., 1., 0.],[1., 1., 0.],[0., 0., 0.]])) 
    C2 = as_matrix(np.array([[1., -1., 0.],[-1., 1., 0.],[0., 0., 1.]]))
    aux_C = [C1, C2]
    
    coef_1 = Expression("1./(2. * (1.0+x[0]) * (1.-2.*x[0]))", degree=1)
    coef_2 = Expression("1./(2. * (1.0 +x[0]))", degree=1)
    coef = [coef_1, coef_2]
    
    param = {"aux_C": aux_C, "nufunc": coef, "Efunc":Expression('x[0]', degree=4),\
             "Afunc":Expression('x[0]', degree=4)} # Define parameteres
    
    # Define BC:
    #-----------------------------
    g1 = [Constant((0.0, 0.5))] # External load
    G = [[g1]]
    
    # Define the problem
    #-----------------------------
    pgd_prob = PGDProblem1(name='temporal_example2D',
                           name_coord=name_coord,
                           modes_info=['U_x', 'Node', 'Vector'],
                           Vs=Vs,
                           dom_fct=create_dom,
                           bc_fct=create_bc,
                           load=G,
                           param=param,
                           rhs_fct=problem_assemble_rhs,
                           lhs_fct=problem_assemble_lhs,
                           probs=prob,
                           seq_fp=seq_prob,
                           PGD_nmax=PGD_nmax)
    
    # Solve the problem
    #-----------------------------
    pgd_prob.stop_fp='norm'
    pgd_prob.tol_fp_it=1e-1
    pgd_prob.tol_abs = 1e-3
    pgd_prob.solve_PGD(_problem='linear')
    
    pgd_solution = pgd_prob.return_PGD()  # as forward model withe evaluate_output e.g for coupling with reliability algorithms
    # pgd_solution.print_info()
    print('PGD Amplitude', pgd_prob.amplitude)
    print(pgd_prob.simulation_info)
    
    return pgd_solution, param

# TEST: PGD result VS FEM result
#==============================================================================  
class PGDproblem(unittest.TestCase):
    
    def setUp(self):
        
        self.E = 1 # Elastic modulus
        self.A = 1 # Amplitude
        self.nu = 0.3 # Poisson ratio
        
    def TearDown(self):
        pass

    def test_solver(self):
        
        # MESH
        #======================================================================
        mesh_0 = RectangleMesh(Point([0,0]), Point([1,1]), 10, 10,'left')
        V_X = VectorFunctionSpace(mesh_0,'Lagrange',2)
        mesh_geo =[mesh_0,'Lagrange',2]

        # [[X1min, X1max, X2min,  X2max,...],[ne1, ne2,...], 'Family', degree]
        mesh_v1 = [[0., 2.],[10],'Lagrange', 1] # Variable 1: Loading amplitude
        mesh_v2 = [[0.5, 1.5],[50],'Lagrange', 1] # Variable 2: Young's modulus
        mesh_v3 = [[0.1, 0.4],[50],'Lagrange', 1] # Variable 3: Poisson's ratio
        input_mesh = [mesh_geo,mesh_v1,mesh_v2,mesh_v3] # Grouping meshes
        
        meshes, Vs = create_meshes(mesh_0,V_X,input_mesh)
        
        # PGD solution
        #======================================================================
        
        # Solve PGD:
        pgd_test, param = main(Vs)
        u_pgd = pgd_test.evaluate(0, [1,2,3], [self.E,self.A,self.nu], 0) # The last zero means to compute displacements
        uvec_pgd = u_pgd.compute_vertex_values()
        
        # FEM solution
        #======================================================================
        mu = 0.5*self.E/(1+self.nu) # 2nd Lame constant
        lmbda = self.E*self.nu/((1-2*self.nu)*(1+self.nu)) # 1st Lame constant (No name lambda because it is the name of a command in python)
        g = Expression(('0.0','0.5'), degree =1)
                       
        # Dirichlet BC:
        bc = create_bc(Vs,0 ,param)
        
        # Neumman BC:
        bnd = create_dom(Vs, param)
        ds = Measure('ds', domain=meshes[0], subdomain_data=bnd[0], subdomain_id=1) # Integral over the boundary   

        u = TrialFunction(Vs[0])
        v = TestFunction(Vs[0])
        d = u.geometric_dimension()
        
        # def epsilon2(u):
        #     return 0.5*(grad(u) + grad(u).T)
        #
        # def sigma(u):
        #     return lmbda*tr(epsilon2(u))*Identity(d) + 2*mu*epsilon2(u)

        def epsilon2(v):
            """ Returns a vector of strains of size (3,1) in the Voigt notation
            layout {eps_xx, eps_yy, gamma_xy} where gamma_xy = 2*eps_xy"""
            return as_vector([v[i].dx(i) for i in range(2)] +
                                    [v[i].dx(j) + v[j].dx(i) for (i, j) in [(0, 1)]])

        def sigma(v):
            # plane strain in E and nu
            alpha = self.E / ((1. + self.nu) * (1 - 2.0 * self.nu))
            C_np = alpha * np.array([[1.0 - self.nu, self.nu, 0.0],
                                     [self.nu, 1.0 - self.nu, 0.0],
                                     [0.0, 0.0, (1.0 - 2.0 * self.nu) / 2.0]])
            C = as_matrix(C_np)
            return C * epsilon2(v)

        rhs = inner(sigma(u),epsilon2(v))*dx  # Right hand side (RHS)
        lhs= dot(g,v)*ds # Left hand side (lhs)
        
        # Solve:
        u = Function(Vs[0])
        solve(rhs==lhs,u,bc[0])
        errorL21 = np.linalg.norm(u_pgd.vector()[:]-u.vector()[:],2)/np.linalg.norm(u.vector()[:],2)
        errorL2 = np.linalg.norm(u_pgd.compute_vertex_values()[:] - u.compute_vertex_values()[:], 2) / np.linalg.norm(u.compute_vertex_values()[:], 2)
        print(errorL21,errorL2)
        self.assertTrue(errorL2<0.03)

if __name__ == '__main__':

    unittest.main()
