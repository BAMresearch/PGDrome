# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import fenics as fe
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import os
# import logging # Show DEBUG message
# import sys # Add path
# sys.path.insert(0, '/home/allona/Bam/2022/PGD/code/General_codes/relpgd')
from relpgd.pgd.solver import PGDProblem1

# from relpgd.forward_models.pgd import PGDModel # Postpro

###############################################################################
################################     INPUT     ################################
###############################################################################

# logging.basicConfig(level=logging.INFO)

# PLATE MATERIAL PROPERTIES
#==============================================================================
# mat = [mat_1, mat_2, ...] = [[type_1,pro1, porp2,..],[type_2,pro1, porp2,..], ...]

# E = 1 # Young's modulus
# nu = Constant(0.3) # Poisson ratio

# Lame parameter
#-----------------------------
# mu = 0.5*E/(1+nu) # 2nd Lame constant
# lmbda = E*nu/((1-2*nu)*(1+nu)) # 1st Lame constant (No name lambda because it is the name of a command in python)

# mat = [[2, 1 ,0.3]] # Plane strain, E = 1, nu = 0.3
#       # TYPE: Material constitutive model.
#                # (1) Elastic 1D: Young , Poisson
#                # (2) Plane strain: Young , Poisson
#                # (3) Plane stress: Young, Poisson

# MESHES
#==============================================================================
# Mesh: Geometry
#-----------------------------
# [[mesh_0, 'Family', degree, matID], [mesh_1, 'Family', degree, matID], ...]

mesh_0 = RectangleMesh(Point([0,0]), Point([1,1]), 10, 10,'left')
V_X = VectorFunctionSpace(mesh_0,'Lagrange',2)

mesh_geo =[mesh_0,'Lagrange',2]

# Mesh: PGD parameters
#-----------------------------
# [[X1min, X1max, X2min,  X2max,...],[ne1, ne2,...], 'Family', degree]
mesh_v1 = [[0., 2.],[10],'Lagrange', 1] # Variable 1: Loading amplitude
mesh_v2 = [[0.5, 1.5],[10],'Lagrange', 1] # Variable 2: Young's modulus
mesh_v3 = [[0.1, 0.4],[10],'Lagrange', 1] # Variable 3: Poisson's ratio

# Grouping meshes
#-----------------------------
input_mesh = [mesh_geo,mesh_v1,mesh_v2,mesh_v3]

# BOUNDARY CONDITIONS:
#==============================================================================

g1 = [Constant((0.0, 0.5))] # External load
# g2 = [Expression('1.0', degree = 4)]
G = [[g1]]

# PGD INPUTS:
#==============================================================================

name_coord = ['X','A','E','P'] # Name variables: i)X: Possition ii)E: Elastic moulus REMOVE --> TO POSTPRO

prob = ['r','s','t','v'] # REMOVE
seq_prob = [0, 1, 2, 3] # REMOVE
PGD_nmax = 20

# DEFINE PARAMETERS:
#==============================================================================
# Plane elastic tensor:
#-----------------------------
C1 = as_matrix(np.array([[1., 1., 0.],[1., 1., 0.],[0., 0., 0.]])) 
C2 = as_matrix(np.array([[1., -1., 0.],[-1., 1., 0.],[0., 0., 1.]]))
aux_C = [C1, C2]

coef_1 = Expression("1./(2. * (1.0+x[0]) * (1.-2.*x[0]))", degree=1)
coef_2 = Expression("1./(2. * (1.0 +x[0]))", degree=1)
coef = [coef_1, coef_2]

param = {"aux_C": aux_C, "nufunc": coef, "Efunc":Expression('x[0]', degree=4)} # Define parameteres

###############################################################################
##############################     COMPUTING     ##############################
###############################################################################

# MESHING AND FINITE ELEMENT MODEL
#==============================================================================

# Initialize:
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
        input_mesh.append(aux_mesh) # Get out of the IF when k==0 is written
        Vs.append(aux_V) # Get out of the IF when k==0 is written

    elif len(input_mesh[k][0]) == 4:
        Pmin = Point(input_mesh[k][0][::2])
        Pmax = Point(input_mesh[k][0][1::2])
        aux_mesh = RectangleMesh(Pmin, Pmax, input_mesh[k][1][0], input_mesh[k][1][1],'left')
        aux_V = VectorFunctionSpace(aux_mesh,input_mesh[k][2],input_mesh[k][3])

    else:
        print("Error: Not well defined")
        
# plot(aux_mesh)
# plt.show()

# BOUNDARY CONDITIONS
#==============================================================================

# Defining DIRICHLET BC:
#-----------------------------

def create_bc(Vs,dom,param):
    
    def BC_ConstDisp(x, on_boundary):
        tol = 1e-5
        return on_boundary and near(x[0], 0, tol) or near(x[1], 0, tol) or near(x[0], 1, tol)  
    
    # on_boundary: Dolfin flags nodes on the boundary with on_boundary
    u0 = Constant((0.,0.))
    bc = DirichletBC(Vs[0],u0,BC_ConstDisp)
    
    return [bc, 0, 0, 0]

# Defining NEUMMAN BC:
#-----------------------------
# Create and initialiye the sub-domain:
# dom = MeshFunction("size_t", meshes[0], meshes[0].topology().dim()-1)
# dom.set_all(0) # initialize the function to zero

# # Create classes for defining parts of the boundaries:
# class BC_Neumman(SubDomain):
#     def inside(self, x, on_boundary):
#         tol = 1e-5
#         return on_boundary and near(x[1], 1, tol) 

# # Initialize sub-domain instance:
# bnd_loadY = BC_Neumman()
# bnd_loadY.mark(dom, 1) # ID of the boundary
# ds = Measure('ds', domain=meshes[0], subdomain_data=bnd, subdomain_id=1) # Integral over the boundary   

# NEUMMAN BC: DOMAIN DEFINITION
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
    
# PROPER GENERALIYED DECOMPOSITION
#==============================================================================
def epsilon(v):
    # Strain: epsilon_ij = 0.5*(du_i/dx_j + du_j/dx_i)
    strain_xx = grad(v)[0,0] #  = u[0].dx(0)
    strain_yy = grad(v)[1,1] #  = u[1].dx(1)
    strain_xy = 0.5* (grad(v)[0,1]+grad(v)[1,0]) #  = 0.5*(u[0].dx(1)+u[1].dx(0))
    
    return as_vector([strain_xx, strain_yy, 2*strain_xy])

# Left Hand Side:
#-----------------------------
def problem_assemble_lhs(fct_F,var_F,Fs,meshes,dom,param,typ,dim):
    
    # Computing R assuming S, T and V is known: 
    if typ == 'r':
        a = assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * param["nufunc"][0] * Fs[3] * dx(meshes[3])) \
            * inner(param["aux_C"][0]*epsilon(fct_F),epsilon(var_F)) * dx(meshes[0]) \
            + assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * param["nufunc"][1] * Fs[3] * dx(meshes[3])) \
            * inner(param["aux_C"][1]*epsilon(fct_F),epsilon(var_F)) * dx(meshes[0])
            
    # Computing S assuming R, T and V is known:
    if typ == 's':
        a = assemble(inner(param["aux_C"][0]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * param["nufunc"][0] * Fs[3] * dx(meshes[3])) \
            * var_F* fct_F * dx(meshes[1]) \
            + assemble(inner(param["aux_C"][1]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * param["nufunc"][1] * Fs[3] * dx(meshes[3])) \
            * var_F* fct_F * dx(meshes[1])
            
    # Computing T assuming R, S and V is known:
    if typ == 't':
        a = assemble(inner(param["aux_C"][0]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[3] * param["nufunc"][0] * Fs[3] * dx(meshes[3])) \
            * var_F * param["Efunc"] * fct_F * dx(meshes[2]) \
            + assemble(inner(param["aux_C"][1]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[3] * param["nufunc"][1] * Fs[3] * dx(meshes[3])) \
            * var_F * param["Efunc"] * fct_F * dx(meshes[2])
            
    # Computing V assuming R, S and T is known:
    if typ == 'v':
        a = assemble(inner(param["aux_C"][0]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * var_F * param["nufunc"][0] * fct_F * dx(meshes[3]) \
            + assemble(inner(param["aux_C"][0]*epsilon(Fs[0]),epsilon(Fs[0])) * dx(meshes[0])) \
            * assemble(Fs[1] * Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * param["Efunc"] * Fs[2] * dx(meshes[2])) \
            * var_F * param["nufunc"][1] * fct_F * dx(meshes[3])

    return a

# Right Hand Side:
#-----------------------------
def problem_assemble_rhs(fct_F, var_F, Fs, meshes, dom, param, G, PGD_func, typ, nE, dim):
    
    ds = Measure('ds', domain=meshes[0], subdomain_data=dom[0], subdomain_id=1) # Integral over the boundary

    l = 0

    # Computing R assuming S, T and V is known: 
    if typ == 'r':
        # for ext in range(len(G[0][0])):
        l += assemble(Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * dx(meshes[3])) \
            * dot(G[0][0][0],var_F) * ds
                
        if nE > 0:
            for old in range(nE):
                l += -assemble(Fs[1] * PGD_func[1][old] * dx(meshes[1])) \
                    * assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])) \
                    * assemble(Fs[3] * param["nufunc"][0] * PGD_func[3][old] * dx(meshes[3])) \
                    * inner(param["aux_C"][0]*epsilon(PGD_func[0][old]),epsilon(var_F)) * dx(meshes[0]) \
                    - assemble(Fs[1] * PGD_func[1][old] * dx(meshes[1])) \
                    * assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])) \
                    * assemble(Fs[3] * param["nufunc"][1] * PGD_func[3][old] * dx(meshes[3])) \
                    * inner(param["aux_C"][1]*epsilon(PGD_func[0][old]),epsilon(var_F)) * dx(meshes[0])
                        
    # Computing S assuming R, T and V is known:
    if typ == 's':
        # for ext in range(len(G[0][1])):
        l += assemble(dot(G[0][0][0],Fs[0]) * ds) \
            * assemble(Fs[2] * dx(meshes[2])) \
            * assemble(Fs[3] * dx(meshes[3])) \
            * var_F * dx(meshes[1])

        if nE > 0:
            for old in range(nE):
                l += -assemble(inner(param["aux_C"][0]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])) \
                    * assemble(Fs[3] * param["nufunc"][0] * PGD_func[3][old] * dx(meshes[3])) \
                    * var_F * PGD_func[1][old] * dx(meshes[1]) \
                    - assemble(inner(param["aux_C"][1]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[2] * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])) \
                    * assemble(Fs[3] * param["nufunc"][1] * PGD_func[3][old] * dx(meshes[3])) \
                    * var_F * PGD_func[1][old] * dx(meshes[1])
                    
                   
    # Computing T assuming R, S and V is known:
    if typ == 't':
        # for ext in range(len(G[0][1])):
        l += assemble(dot(G[0][0][0],Fs[0]) * ds) \
            * assemble(Fs[1] * dx(meshes[1])) \
            * assemble(Fs[3] * dx(meshes[3])) \
            * var_F * dx(meshes[2])

        if nE > 0:
            for old in range(nE):
                l += -assemble(inner(param["aux_C"][0]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[1]*PGD_func[1][old] *dx(meshes[1])) \
                    * assemble(Fs[3] * param["nufunc"][0] * PGD_func[3][old] * dx(meshes[3])) \
                    * var_F * param["Efunc"] * PGD_func[2][old] * dx(meshes[2]) \
                    - assemble(inner(param["aux_C"][1]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[1]*PGD_func[1][old] *dx(meshes[1])) \
                    * assemble(Fs[3] * param["nufunc"][1] * PGD_func[3][old] * dx(meshes[3])) \
                    * var_F * param["Efunc"] * PGD_func[2][old] * dx(meshes[2])

    # Computing V assuming R, S and T is known:
    if typ == 'v':
        # for ext in range(len(G[0][1])):
        l += assemble(dot(G[0][0][0],Fs[0]) * ds) \
            * assemble(Fs[1] * dx(meshes[1])) \
            * assemble(Fs[2] * dx(meshes[2])) \
            * var_F * dx(meshes[3])

        if nE > 0:
            for old in range(nE):
                l += -assemble(inner(param["aux_C"][0]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[1]*PGD_func[1][old] *dx(meshes[1])) \
                    * assemble(Fs[2]*param["Efunc"]*PGD_func[2][old] *dx(meshes[2])) \
                    * var_F * param["nufunc"][0] * PGD_func[3][old] * dx(meshes[3]) \
                    - assemble(inner(param["aux_C"][1]*epsilon(PGD_func[0][old]),epsilon(Fs[0])) * dx(meshes[0])) \
                    * assemble(Fs[1]*PGD_func[1][old] *dx(meshes[1])) \
                    * assemble(Fs[2]*param["Efunc"]*PGD_func[2][old] *dx(meshes[2])) \
                    * var_F * param["nufunc"][1] * PGD_func[3][old] * dx(meshes[3])
                            
    return l

# SOLVING PGD PROBLEM
#==============================================================================
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
pgd_prob.solve_PGD(_type='normal', _problem='linear')

# pgd_prob.max_fp_it = 30
# pgd_prob.tol_fp_it = 1e-6
# pgd_prob.tol_abs = 1e-7
# pgd_prob.max_fp_it = 100
# pgd_prob.tol_fp_it = 1e-8
# pgd_prob.tol_abs = 1e-9

# pgd_s = pgd_prob.return_PGD()  # as PGD class instance
# pgd_s.print_info()
pgd_solution = pgd_prob.return_PGDModel()  # as forward model withe evaluate_output e.g for coupling with reliability algorithms

pgd_solution.print_info()

###############################################################################
###############################     POSTPRO     ###############################
###############################################################################

# for m in range(pgd_solution.numModes):
#     ax1.plot(meshes[0].coordinates()[:],pgd_solution_normal.mesh[0].attributes[0].interpolationfct[m].vector()[:])

# Plot modes of the variables:
#-----------------------------
ax = [[], [], [], []]
fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, pgd_solution.num_pgd_var-1)
for k in range(1,pgd_solution.num_pgd_var):
    for m in range(pgd_solution.numModes):    
        pgd_dof = pgd_solution.mesh[k].attributes[0].interpolationfct[m].vector()[:]
        x_dim = pgd_solution.mesh[k].attributes[0].interpolationfct[m].function_space().mesh().geometry().dim()
        x_dof = pgd_solution.mesh[k].attributes[0].interpolationfct[m].function_space().tabulate_dof_coordinates().reshape((-1, x_dim))[:,0]
        zipped_data = zip(x_dof, pgd_dof)
        temp = sorted(zipped_data, key =lambda x: x[0])
        X, Y = map(list, zip(*temp))
        ax[k-1].plot(X, Y)
    ax[k-1].set_title(name_coord[k])

# for m in range(pgd_solution.numModes):    
#     pgd_dof = pgd_solution.mesh[3].attributes[0].interpolationfct[m].vector()[:]
#     x_dim = pgd_solution.mesh[3].attributes[0].interpolationfct[m].function_space().mesh().geometry().dim()
#     x_dof = pgd_solution.mesh[3].attributes[0].interpolationfct[m].function_space().tabulate_dof_coordinates().reshape((-1, x_dim))[:,0]
#     zipped_data = zip(x_dof, pgd_dof)
#     temp = sorted(zipped_data, key =lambda x: x[0])
#     X, Y = map(list, zip(*temp))
#     plt.plot(X, Y,label='%s data' % m)
    
# plt.legend()
# plt.show()
     

# Plot the results:
#-----------------------------
import matplotlib.tri as tri

u_pgd = pgd_solution.evaluate(0, [1,2,3], [1,1,0.3], 0) # The last zero means to compute displacements
uuu = u_pgd.compute_vertex_values()
aaa=np.array([uuu[:121], uuu[121:]])

X0 = mesh_0.coordinates()
T = mesh_0.cells()

Xnew = X0 + aaa.T
scalars = uuu[121:]

triangulation_0 = tri.Triangulation(X0[:,0], X0[:,1], T)
plt.triplot(triangulation_0, '-k')
triangulation_1 = tri.Triangulation(Xnew[:,0], Xnew[:,1], T)
plt.triplot(triangulation_1, '--r')

plt.tricontourf(triangulation_1, scalars)
plt.colorbar()
plt.show()