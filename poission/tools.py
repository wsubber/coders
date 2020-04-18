#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:45:20 2020

@author: 212731466
"""


from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import matplotlib.tri as tri
import pandas as pd
import GPy
#%%
def get_flux_fromGPy(u_old,u_new,gp_model):

    V0 = u_old.function_space()
    submesh0 = V0.mesh()
    lambda0=u_old.vector()    
    ub0_old, X0, lambda0_vector_index=get_interface_values_from_vector(0.5,V0,submesh0,lambda0)
    
    lambda0=u_new.vector()
    ub0_new, X0, lambda0_vector_index=get_interface_values_from_vector(0.5,V0,submesh0,lambda0)
    
    ub0=0.5*(ub0_old+ub0_new)
    lambda0_array=np.zeros(submesh0.num_vertices())

    Xnew=np.array([X0[:,1], ub0]).T.reshape(-1,2)
    
    predicted_flux=gp_model.predict(Xnew)
    predicted_flux=predicted_flux[0]
    
    lambda0_array[lambda0_vector_index]=predicted_flux.reshape(-1,)
    lambda0.set_local(lambda0_array)
    
    flux0=Vector2Function(V0,lambda0)
    
    return flux0




#%% get interface values working with vectors
def get_interface_values_from_vector(xb,V0,mesh0,u0_vec):
    InterfaceBoundary = AutoSubDomain(lambda x, on_bnd: near(x[0], xb) and on_bnd)
    bc_Iinterface = DirichletBC(V0, 1, InterfaceBoundary)

    interface_indecater = Function(V0)
    bc_Iinterface.apply(interface_indecater.vector())
    index0=interface_indecater.vector().get_local().astype(bool)
    # interface u
    ub=u0_vec.get_local()[index0]
    X0=V0.tabulate_dof_coordinates()[index0,:]

    return ub,X0,index0


#%% # get interface values working with functions
def get_interface_values_from_function(xb,V0,mesh0,u0_fun):
    InterfaceBoundary = AutoSubDomain(lambda x, on_bnd: near(x[0], xb) and on_bnd)
    bc_Iinterface = DirichletBC(V0, 1, InterfaceBoundary)

    interface_indecater = Function(V0)
    bc_Iinterface.apply(interface_indecater.vector())
    index0=interface_indecater.compute_vertex_values(mesh0).astype(bool)
    # interface u
    ub=u0_fun.compute_vertex_values(mesh0)[index0]
    X0=mesh0.coordinates()[index0,:]

    return ub,X0,index0


def compute_errors(u_e, u):
    """Compute various measures of the error u - u_e, where
    u is a finite element Function and u_e is an Expression."""

    # Get function space
    V = u.function_space()

    # Explicit computation of L2 norm
    error = (u - u_e)**2*dx
    E1 = sqrt(abs(assemble(error)))

    # Explicit interpolation of u_e onto the same space as u
    u_e_ = interpolate(u_e, V)
    error = (u - u_e_)**2*dx
    E2 = sqrt(abs(assemble(error)))

    # Explicit interpolation of u_e to higher-order elements.
    # u will also be interpolated to the space Ve before integration
    Ve = FunctionSpace(V.mesh(), 'P', 5)
    u_e_ = interpolate(u_e, Ve)
    error = (u - u_e)**2*dx
    E3 = sqrt(abs(assemble(error)))

    # Infinity norm based on nodal values
    u_e_ = interpolate(u_e, V)
    E4 = abs(u_e_.vector().get_local() - u.vector().get_local()).max()

    # L2 norm
    E5 = errornorm(u_e, u, norm_type='L2', degree_rise=3)

    # H1 seminorm
    E6 = errornorm(u_e, u, norm_type='H10', degree_rise=3)

    # Collect error measures in a dictionary with self-explanatory keys
    errors = {'u - u_e': E1,
              'u - interpolate(u_e, V)': E2,
              'interpolate(u, Ve) - interpolate(u_e, Ve)': E3,
              'infinity norm (of dofs)': E4,
              'L2 norm': E5,
              'H10 seminorm': E6}

    return errors, E5

def compute_convergence_rates(u_e, f, u_D, kappa,
                              max_degree=3, num_levels=5):
    "Compute convergences rates for various error norms"

    h = {}  # discretization parameter: h[degree][level]
    E = {}  # error measure(s): E[degree][level][error_type]

    # Iterate over degrees and mesh refinement levels
    degrees = range(1, max_degree + 1)
    for degree in degrees:
        n = 8  # coarsest mesh division
        h[degree] = []
        E[degree] = []
        for i in range(num_levels):
            h[degree].append(1.0 / n)
            u = solver(kappa, f, u_D, n, n, degree, linear_solver='direct')
            errors = compute_errors(u_e, u)
            E[degree].append(errors)
            print('2 x (%d x %d) P%d mesh, %d unknowns, E1 = %g' %
              (n, n, degree, u.function_space().dim(), errors['u - u_e']))
            n *= 2

    # Compute convergence rates
    from math import log as ln  # log is a fenics name too
    etypes = list(E[1][0].keys())
    rates = {}
    for degree in degrees:
        rates[degree] = {}
        for error_type in sorted(etypes):
            rates[degree][error_type] = []
            for i in range(1, num_levels):
                Ei = E[degree][i][error_type]
                Eim1 = E[degree][i - 1][error_type]
                r = ln(Ei / Eim1) / ln(h[degree][i] / h[degree][i - 1])
                rates[degree][error_type].append(round(r, 2))

    return etypes, degrees, rates


def Vector2Function(V,b):
     u= Function(V)
     if isinstance(b,Vector):
         values=b.get_local()
     else:
         values=b
     u.vector().set_local(values)
     return(u)
 
def Flux_old(submesh0,u0,v0,ds):
    n0 = FacetNormal(submesh0)
    m0 = dot(grad(u0), n0)*v0*ds(0)
    flux_vector = assemble(m0)
    return flux_vector


def Flux(u, kappa):
#    "Return -kappa*grad(u) projected into same space as u"
    V = u.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'P', degree)
    flux_u = project(-kappa*grad(u), W)
    return flux_u

def Flux_over_domain(u):
    V = u.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'P', degree)

    grad_u = project(grad(u), W)
    k=1
    flux_u = project(-k*grad(u), W)

    return flux_u

def Plot_solution(mesh,u):
    plt.figure()
    triang = tri.Triangulation(*mesh.coordinates().reshape((-1, 2)).T,
                                   triangles=mesh.cells())
    plt.tricontourf(triang, u.compute_vertex_values())
    plt.colorbar()
    plt.show()


def print_bc(mesh,bc,V):
    # Print all vertices that belong to the boundary parts
    for x in mesh.coordinates():
        if bx0.inside(x, True): print('%s is on x = 0' % x)
        if bx1.inside(x, True): print('%s is on x = 1' % x)
        if by0.inside(x, True): print('%s is on y = 0' % x)
        if by1.inside(x, True): print('%s is on y = 1' % x)

    # Print the Dirichlet conditions
    print('Number of Dirichlet conditions:', len(bcs))
    if V.ufl_element().degree() == 1:  # P1 elements
        d2v = dof_to_vertex_map(V)
        coor = mesh.coordinates()
        for i, bc in enumerate(bcs):
            print('Dirichlet condition %d' % i)
            boundary_values = bc.get_boundary_values()
            for dof in boundary_values:
                print('   dof %2d: u = %g' % (dof, boundary_values[dof]))
                if V.ufl_element().degree() == 1:
                    print('    at point %s' %
                          (str(tuple(coor[d2v[dof]].tolist()))))
    return 0

def check_con():
    def get_error(u,u0,u1,V):
        corr=V.tabulate_dof_coordinates()
        
        u_xy = np.zeros([corr.shape[0], 1])
        u0_xy= np.zeros([corr.shape[0], 1])
        u1_xy= np.zeros([corr.shape[0], 1])
        for idx,ip_xy in enumerate(corr):
            u_xy[idx]=u(ip_xy)
            if ip_xy[0] < 0.5 :
                u0_xy[idx]=u0(ip_xy)
            elif ip_xy[0] > 0.5 :
                u1_xy[idx]=u1(ip_xy)
            elif ip_xy[0]== 0.5 :
                u0_xy[idx]=(u0(ip_xy)+u1(ip_xy))/2.0
                u1_xy[idx]=(u0(ip_xy)+u1(ip_xy))/2.0
        
        return u_xy,u0_xy,u1_xy
    
    u_xy,u0_xy,u1_xy=get_error(u,u0,u1,V)
    
    u_from0=u.copy(deepcopy=True)
    u_from1=u.copy(deepcopy=True)
    v=u.copy(deepcopy=True)
    u_from0.vector().set_local(u0_xy)
    u_from1.vector().set_local(u1_xy)
    
    
    v_value=(u.vector().get_local()-u_from0.vector().get_local()-u_from1.vector().get_local())
    v.vector().set_local(v_value)
    plt.figure()
    triang = tri.Triangulation(*mesh.coordinates().reshape((-1, 2)).T,
                                       triangles=mesh.cells())
    plt.tricontourf(triang, v.compute_vertex_values())
    plt.colorbar()
    plt.show()
    print(n, norm(v))
    #2 0.015145077118060875
    #4 0.004791356350758525
    #8 0.002916102309183629
    #16 0.0027945523473865913
    #32 0.0027998199569695655
    #64 0.0028036303409515926
    #128 0.0028047880160956542
    #256 0.002805076834027968
    #%%
    
    
    
    n = [2, 4, 8, 16, 32, 64, 128, 256]
    
    for in_ in n:
        submesh0 = RectangleMesh(Point(0.0, 0.0), Point(0.5,1.0), in_, in_)
        print(submesh0.hmax())
    
    x = [0.5590169943749475,
    0.2795084971874737,
    0.13975424859373686,
    0.06987712429686843,
    0.034938562148434216,
    0.017469281074217108,
    0.008734640537108554,
    0.004367320268554277]
    
    y = [0.015145077118060875, 0.004791356350758525, 0.002916102309183629, 
         0.0027945523473865913, 0.0027998199569695655, 0.0028036303409515926,
         0.0028047880160956542, 0.002805076834027968]
    plt.loglog(x, y)
    plt.loglog(y, y)    