#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:29:36 2020

@author: 212731466
"""

from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import matplotlib.tri as tri
from tools import *
from solvers import *
import pandas as pd
import matplotlib
from mshr import Rectangle, Circle, generate_mesh
#%% now using u1 get ub from subdomain 1 to subdomain 0

n=32
 
df = pd.read_csv('mesh_with_holls_05.csv')
no_circles=len(df)
cir0={}
cir1={}
r = 0.05
icir0=0
icir1=0
for idx,row in df.iterrows():
    if  row['x']+r < 0.5:
        cir0[icir0]= Circle(Point(row['x'],row['y']), r)
        icir0 = icir0 + 1
    else:
        cir1[icir1]= Circle(Point(row['x'],row['y']), r)
        icir1= icir1 + 1

#%%
submesh0 = Rectangle(Point(0.0, 0.0), Point(0.5,1.0))
submesh1 = Rectangle(Point(0.5, 0.0), Point(1.0,1.0))

for idx,row in cir0.items():
    submesh0 = submesh0 + row

for idx,row in cir1.items():
    submesh1 = submesh1 + row
    
#%%    
for idx,row in cir0.items():
    submesh0.set_subdomain(idx+1,row)

for idx,row in cir1.items():
    submesh1.set_subdomain(idx+1,row)
    
# Create mesh
mesh0 = generate_mesh(submesh0, n)
mesh1 = generate_mesh(submesh1, n) 


V0 = FunctionSpace(mesh0,'CG',1)
u_avaraged=Function(V0)
for i in range(20):
    flux0,u0=Dsolve(mesh0,u_avaraged)
    u1=Nsolve(mesh1, flux0)
    u_avaraged=0.5*(u1+u0)
 
plot(u0)