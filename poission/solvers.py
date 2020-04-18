#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:35:15 2020

@author: 212731466
"""


from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import matplotlib.tri as tri
from tools import *
import pandas as pd

def generate_samples(submesh0, submesh1,nsamples=10,f=1):
    V0 = FunctionSpace(submesh0,'CG',1)
    u_avaraged=Function(V0)
    
    df=pd.DataFrame()
    df_k=pd.DataFrame()
        
    for isample in range(nsamples):
        flux0,u0=Dsolve(submesh0,u_avaraged,f)
        u1=Nsolve(submesh1, flux0,f)
        u_avaraged=0.5*(u1+u0)
        
        lamb,X0,index0=get_interface_values_from_function(0.5,V0,submesh0,flux0)
    
        u0_,X0_,index0_=get_interface_values_from_function(0.5,V0,submesh0,u0)
        df_k['x']=X0[:,1]
        df_k['ub']=u0_
        df_k['lambda']=lamb
    
        df=df.append(df_k,ignore_index=True)
    
    return df

def Fsolve(mesh,Plot=False):
    
    V = FunctionSpace(mesh,'CG',1)
    u = TrialFunction(V)
    v = TestFunction(V)
    
    def Dirichlet_boundary(x, on_boundary):
        return on_boundary and near(x[0],0.0) or near(x[1],1.0) or near(x[1],0.0) or near(x[0],1.0)
    
    bc = DirichletBC(V, 0.0,Dirichlet_boundary)
    
    f = Constant(1.0)
    
    a = inner(grad(u), grad(v))*dx
    L =  f*v*dx
    
    A = assemble(a)
    b = assemble(L)
    
    bc.apply(A, b)
    
    u = Function(V)
    solve(A, u.vector(), b)
    
    if Plot:
        Plot_solution(mesh,u)

    return u

 
def Dsolve(submesh0,u1,f0=1.0,Plot=False):
    
    V0 = FunctionSpace(submesh0,'CG',1)
    u0 = TrialFunction(V0)
    v0 = TestFunction(V0)
    
    def Dirichlet_boundary0(x, on_boundary):
        return on_boundary and  near(x[1],1.0) or near(x[1],0.0)
    
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)

    
    class Interface(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.5)
            
    
    facets = MeshFunction("size_t", submesh0, submesh0.topology().dim()-1, 0)
    facets.set_all(1)
    interface= Interface()
    interface.mark(facets, 0)
    left = Left()
    left.mark(facets, 2)
    #ds = Measure("ds")[facets]
    ds = Measure('ds', subdomain_data=facets)    
    tag=0
    
    class K(UserExpression):
        def __init__(self, facets,tag, u1, **kwargs):
            super().__init__(**kwargs)
            self.facets = facets
            self.u1 = u1
            self.tag = tag
        def eval_cell(self, values, x, cell):
            if near(x[0], 0.5):
                values[0]=u1([x[0],x[1]])
        def value_shape(self):
            return ()
    

    kappa = K(facets,tag,u1)
    
    bcinterface = DirichletBC(V0, kappa,interface)
    
    
    
    bc0 = DirichletBC(V0, 0.0,Dirichlet_boundary0)
    
    g_L = Expression("0.1*sin(x[1])",degree=2) 
    
    a0 = inner(grad(u0), grad(v0))*dx
    L0 =  f0*v0*dx + g_L*v0*ds(2)
    
    A0 = assemble(a0)
    b0 = assemble(L0)
    
    bc0.apply(A0, b0)
    bcinterface.apply(A0, b0)
    
    u0 = Function(V0)
    solve(A0, u0.vector(), b0)
    
    if Plot:
        Plot_solution(submesh0,u0)
    # compute flux    
    g=Flux(u0, 1)
    n0 = FacetNormal(submesh0)
    m0 = dot(g, n0)*v0*ds(0)
    flux_vector = assemble(m0)
              
    flux_function=Vector2Function(V0,flux_vector)
       
    return flux_function, u0
#%%
# subdomain 1
def Nsolve(submesh1,flux0,f1=1.0,Plot=False):
    V1 = FunctionSpace(submesh1,'CG',1)
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    
    def Dirichlet_boundary1(x, on_boundary):
        return on_boundary and near(x[1],1.0) or near(x[1],0.0)
    
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)
    
    class Interface(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.5)
    facets = MeshFunction("size_t", submesh1, submesh1.topology().dim()-1, 0)
    facets.set_all(0)
    interface= Interface()
    tag = 1
    interface.mark(facets, tag)
    right = Right()
    right.mark(facets, 2)
    
    
    ds = Measure('ds', subdomain_data=facets)    
    
    bc1 = DirichletBC(V1, 0.0,Dirichlet_boundary1)
    bc2 = DirichletBC(V1, 0.1,right)
#    f1 = Constant(1.0)
    
    a1 = inner(grad(u1), grad(v1))*dx
     
     
    #flux0.set_allow_extrapolation(True)
    class Compute_Flux(UserExpression):
        def __init__(self, facets,tag, flux0, **kwargs):
            super().__init__(**kwargs)
            self.facets = facets
            self.flux0 = flux0
            self.tag = tag
        def eval_cell(self, values, x, cell):
            if near(x[0], 0.5):
                values[0]=flux0([x[0],x[1]])
        def value_shape(self):
            return ()
    
    flux1 = Compute_Flux(facets,tag,flux0)
    
    L1 =  f1*v1*dx  + flux1*v1*ds(1)
    
    A1 = assemble(a1)
    b1 = assemble(L1)
    
    bc1.apply(A1, b1)
    bc2.apply(A1, b1)    
    u1 = Function(V1)
    solve(A1, u1.vector(), b1)
    
    if Plot:
        Plot_solution(submesh1,u1)
    
    return u1

def Dsolve1D(nelem = 100,f=1.0):

    mesh = UnitIntervalMesh(nelem)
    
    V = FunctionSpace(mesh,'CG',1)
    
    # Define Dirichlet boundary condition
    u_D = Constant(0.0)
    
    def Dirichlet_boundary(x, on_boundary):
        return on_boundary
    
    bc = DirichletBC(V, u_D, Dirichlet_boundary)
    
    # Define trial and test spaces
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # the bilinear form
    a = dot(grad(u), grad(v))*dx
    # and the linear form
    L = f*v*dx
    
    u = Function(V)
    # Solve the problem for u, with the given boundary conditions.
    solve(a == L, u, bc)
    
    plot(u)
    g=Flux(u, 1)
    n = FacetNormal(mesh)
    m = dot(g, n)*v*ds # this will do integration only at the bc 
    flux_vector = assemble(m)
     
    #flux_function=Vector2Function(V0,flux_vector)
    
    flux_u=Flux_over_domain(u)
    
    plot(flux_u, title='flux field')

#    flux_x, flux_y = flux_u.split(deepcopy=True)  # extract components
 #   plot(flux_x, title='x-component of flux (-kappa*grad(u))')
#    plot(flux_y, title='y-component of flux (-kappa*grad(u))')

    return u,flux_u

