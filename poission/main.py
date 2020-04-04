#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:13:29 2020

@author: 212731466
"""
#%%
from dolfin import *
import numpy as np

from solvers import *
from utlity import *
#%% solve the global problem
# set mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(2.0,1.0), 30, 30)
# set Dirichlet BC value
u0=0
left=0.0
top=1.0
right=2.0
bottom=0.0
u=Dirichlet(mesh,u0,left,top,right,bottom)
# now the mesh is decomposed into
# mesh = mesh0 + mesh1
#%% solve subdomain 0
mesh0 = RectangleMesh(Point(0.0, 0.0), Point(1.0,1.0), 30, 30)
# set Dirichlet BC value
initial=0
left=0.0
top=1.0
right=1.0
bottom=0.0
u0 = Dirichlet(mesh0,initial,left,top,right,bottom)
lambda0 = GetFlux(mesh0,u0)
#%% second ddomain
mesh1 = RectangleMesh(Point(1.0, 0), Point(2,1.0), 30, 30)
u1=Neumann(mesh1,lambda0)

#%% get values on the boundary
u0=Dirichlet(mesh0,0.0,left,top,right,bottom,True,u1)
lambda0 = GetFlux(mesh0,u0)
#%% second iteration
u1=Neumann(mesh1,lambda0)
#%%


n0 = norm(u0, 'H1', mesh0)

n1 = norm(u1, 'H1', mesh1)

print(n0-n1)

#%%
