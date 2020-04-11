#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:13:29 2020

@author: 212731466
"""
#%%
from dolfin import *
import numpy as np
parameters['allow_extrapolation'] = True
from solvers import *
from utlity import *
from mshr import *
#%% solve the global problem
# set mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(2.0,1.0), 20, 20)
V = FunctionSpace(mesh, "CG", 1)
u=Direct(mesh)

#%%

# now the mesh is decomposed into
# mesh = mesh0 + mesh1
mesh_c = RectangleMesh(Point(0.0, 0.0), Point(2.0,1.0), 30, 30)
V_c = FunctionSpace(mesh_c, 'CG', 1)

#%% define firest domain  mesh0
mesh0 = RectangleMesh(Point(0.0, 0.0), Point(1.0,1.0), 10, 10)
V0 = FunctionSpace(mesh0, 'CG', 1)
# define second domain mesh 1
mesh1 = RectangleMesh(Point(1.0, 0), Point(2,1.0), 10, 10)
V1 = FunctionSpace(mesh1, 'CG', 1)
# set initeal Dirichlet BC
initial=0.0
u0 = Dirichlet(mesh0,initial)
# get flux
lambda0_vec, lambda0_fun= GetFlux(mesh0,u0)

for iiter in range(2):
    #% solve Neumann in second domsin
    u1=Neumann(mesh1,lambda0_fun)
    # get values on the boundary
    u0=Dirichlet(mesh0,u1)
    # get flux
    lambda0_vec, lambda0_fun = GetFlux(mesh0,u0)
#%%



plt.figure()
plot(u0)
plot(u1)
plt.show()

n0 = norm(u0)

n1 = norm(u1)

n = norm(u)

print("n",n)
print("n0",n0)
print("n1",n1)

#%%
if 0:
    mesh0 = RectangleMesh(Point(0.0, 0.0), Point(1.0,1.0), 2, 2)
    mesh1 = RectangleMesh(Point(0.0, 0.0), Point(1.0,1.0), 20, 20)
    V0 = FunctionSpace(mesh0, "CG", 1)
    V1 = FunctionSpace(mesh1, "CG", 1)

    # set initeal Dirichlet BC
    initial=0.0
    u0 = Dirichlet(mesh0,initial)
    u1 = Dirichlet(mesh1,initial)

    from fenicstools import interpolate_nonmatching_mesh

    u2 = interpolate_nonmatching_mesh(u0, V1)

    #%%
    parameters['allow_extrapolation'] = True
    mesh1 = UnitSquareMesh(16, 16)
    V1 = FunctionSpace(mesh1, 'CG', 2)
    u1 = interpolate(Expression("sin(pi*x[0])*cos(pi*x[1])",degree=2), V1)
    # Create a new _different_ mesh and FunctionSpace
    mesh2 = UnitSquareMesh(10, 10)
    x = mesh2.coordinates()
    x[:, :] = x[:, :] * 0.5 + 0.25
    V2 = FunctionSpace(mesh2, 'CG', 1)
    u2 = interpolate_nonmatching_mesh(u1, V2)
    u3 = interpolate_nonmatching_mesh(u2, V1)