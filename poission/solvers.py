#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 09:37:55 2020

@author: 212731466
"""
#%%
from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import matplotlib.tri as tri
from utlity import *
import GPy
matplotlib.rc('font',size=16)
matplotlib.rcParams["font.family"] = "Times New Roman"

def Dirichlet(mesh,u1):

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    if isinstance(u1,float) or isinstance(u1,int):
        u1 = Function(V)


    def Dirichlet_boundary(x, on_boundary):
            return on_boundary and near(x[0],0.0)  or near(x[1],1.0) or near(x[1],0.0)
    u0 = 0.0
    bc = DirichletBC(V, u0,Dirichlet_boundary)


    class Interface(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)
    interface= Interface()
    facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    facets.set_all(0)
    tag = 5
    interface.mark(facets, tag)

    class K(UserExpression):
        def __init__(self, facets,tag, u1, **kwargs):
            super().__init__(**kwargs)
            self.facets = facets
            self.u1 = u1
            self.tag = tag
        def eval_cell(self, values, x, cell):
            if near(x[0], 1.0):
                values[0]=u1([x[0],x[1]])
        def value_shape(self):
            return ()

    kappa = K(facets,tag,u1)
    bcinterface = DirichletBC(V, kappa,interface)

    f = Constant(1.0)

    a = inner(grad(u), grad(v))*dx
    L =  f*v*dx

    A = assemble(a)
    b = assemble(L)

    bc.apply(A, b)

    bcinterface.apply(A, b)

    u = Function(V)
    solve(A, u.vector(), b)


    triang = tri.Triangulation(*mesh.coordinates().reshape((-1, 2)).T,
                               triangles=mesh.cells())

    plt.figure()
    plt.tricontourf(triang, u.compute_vertex_values())
    plt.colorbar()
    plt.show()

    return u

def Neumann(mesh,lamda):

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    def Dirichlet_boundary1(x, on_boundary):
            return on_boundary and near(x[0],2.0) or near(x[1],0) or near(x[1],1)

    bc = DirichletBC(V, 0.0,Dirichlet_boundary1)
    f = Constant(1.0)

    class Interface(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)


    interface = Interface()
    boundaries = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
    boundaries.set_all(0)
    tag=5
    interface.mark(boundaries, tag)

    a = inner(grad(u), grad(v))*dx

    if isinstance(lamda, Vector):
        g = Vector2Function(V,lamda)
    elif isinstance(lamda, Function):
        g = project(lamda, V)

    facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    facets.set_all(0)
    interface.mark(facets, tag)
    ds = Measure("ds")[facets]

    L =  f*v*dx + g*v*ds(1)

    A = assemble(a)
    b = assemble(L)

    bc.apply(A, b)
    #<-intefac
    u = Function(V)
    solve(A, u.vector(), b)
    triang = tri.Triangulation(*mesh.coordinates().reshape((-1, 2)).T,
                               triangles=mesh.cells())

    plt.figure()
    plt.tricontourf(triang, u.compute_vertex_values())
    plt.colorbar()
    plt.show()

    return u

def Direct(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    def Dirichlet_boundary(x, on_boundary):
            return on_boundary and near(x[0],0.0) or near(x[0],2.0) or near(x[1],1.0) or near(x[1],0.0)
    u0 = 0.0
    bc = DirichletBC(V, u0,Dirichlet_boundary)

    f = Constant(1.0)

    a = inner(grad(u), grad(v))*dx
    L =  f*v*dx

    A = assemble(a)
    b = assemble(L)

    bc.apply(A, b)

    u = Function(V)
    solve(A, u.vector(), b)


    triang = tri.Triangulation(*mesh.coordinates().reshape((-1, 2)).T,
                               triangles=mesh.cells())

    plt.figure()
    plt.tricontourf(triang, u.compute_vertex_values())
    plt.colorbar()
    plt.show()

    return u
