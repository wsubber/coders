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
#%% now using u1 get ub from subdomain 1 to subdomain 0
matplotlib.rc('font',size=16)
matplotlib.rcParams["font.family"] = "Times New Roman"
omega = 1.0
u_e = Expression('sin(omega*pi*x[0])*sin(omega*pi*x[1])',
                 degree=6, omega=omega)
f = 2*omega**2*pi**2*u_e


def DDM_solver(n):
    submesh0 = RectangleMesh(Point(0.0, 0.0), Point(0.5,1.0), n, n)
    submesh1 = RectangleMesh(Point(0.5, 0.0), Point(1.0,1.0), n, n)    
    V0 = FunctionSpace(submesh0,'CG',1)
    u_avaraged=Function(V0)
    for i in range(20):
        flux0,u0=Dsolve(submesh0,u_avaraged,f)
        u1=Nsolve(submesh1, flux0,f)
        u_avaraged=0.5*(u1+u0)
    return u0,u1



num_levels=5

n = 8  # coarsest mesh division
h = []
E = []
L2norm0 = []
L2norm1 = []
for i in range(num_levels):
    h.append(1.0 / n)
    # DDM solver
    u0, u1 = DDM_solver(n)
    errors_0,E5_0 = compute_errors(u_e, u0)
    L2norm0.append(E5_0)

    errors_0,E5_1 = compute_errors(u_e, u1)
    L2norm1.append(E5_1)
    print('2 x (%d x %d), %d unknowns, E1 = %g' %
      (n, n, u0.function_space().dim(), errors_0['u - u_e']))
        
    n *= 2

#%%
from matplotlib import rc

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
h=np.array(h)
plt.figure()
plt.loglog(h, np.array(L2norm0)+np.array(L2norm1),'-o',label='DDM')
plt.xlabel('$h$')
plt.ylabel('$L_2$')
plt.loglog(h, h**2,'rs:',label='$O(h^2)$')
plt.legend()
plt.savefig('con.eps', format='eps')
plt.show()