#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:10:18 2020

@author: 212731466
"""
#%%

from fenics import *
from mshr import Rectangle, Circle, generate_mesh
import pandas as pd
import numpy as np

# read data
df = pd.read_csv('mesh_with_holls_05.csv')
no_circles=len(df)
cir={}
r = 0.05
for idx,row in df.iterrows():
    cir[idx]= Circle(Point(row['x'],row['y']), r)

#%%

rec = Rectangle(Point(0.0, 0.0), Point(1.0,1.0))

domain = rec

for idx,row in cir.items():
    domain = domain + row

    
#%%    
for idx,row in cir.items():
    domain.set_subdomain(idx+1,row)
 
# Create mesh
mesh = generate_mesh(domain, 128)
 
# Define subdomain markers and integration measure
markers = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())
dx = Measure('dx', domain=mesh, subdomain_data=markers)
plot(markers)
#%%
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)
# Initialize sub-domain instances
left = Left()
top = Top()
right = Right()
bottom = Bottom()    



boundaries = MeshFunction('size_t',mesh,mesh.topology().dim()-1, mesh.domains())
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)
ds = Measure('ds',domain=mesh, subdomain_data=boundaries)
# Define boundary condition
 
V = FunctionSpace(mesh,'CG',1)
u = TrialFunction(V)
v = TestFunction(V)

bcs = [DirichletBC(V, 0.0, boundaries, 2),
       DirichletBC(V, 0.1, boundaries, 3),
       DirichletBC(V, 0.0, boundaries, 4)]

        

g_L = Expression("0.1*sin(x[1])",degree=2)
f = Constant(1.0)

 
kappa = np.ones((no_circles+1, )) * 0.01
kappa[0] = 1.0

a = kappa[0]* inner(grad(u), grad(v))*dx(0)
L =  f*v*dx(0)

for idx in range(no_circles):
    a = a + kappa[idx+1]* inner(grad(u), grad(v))*dx(idx+1)
    L = L + f*v*dx(idx+1)

L = L +g_L*v*ds(1)
u = Function(V)

solve(a == L, u, bcs)
plot(u)
 