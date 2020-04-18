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
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
import pandas as pd

n =128
submesh0 = RectangleMesh(Point(0.0, 0.0), Point(0.5,1.0), n, n)
submesh1 = RectangleMesh(Point(0.5, 0.0), Point(1.0,1.0), n, n)
#%% Genrate data from GP
V0 = FunctionSpace(submesh0,'CG',1)
V1 = FunctionSpace(submesh1,'CG',1)
#%% Train a GP


omega = 1.0
u_e = Expression('sin(omega*pi*x[0])*sin(omega*pi*x[1])',
                 degree=6, omega=omega)
f = 2*omega**2*pi**2*u_e

nsamples=10
df = generate_samples(submesh0, submesh1,10,f)
 
X = np.zeros((df.shape[0],2))
X[:,0] = df['x'].values
X[:,1] = df['ub'].values
Y = df['lambda'].values.reshape(df.shape[0],1)


kernel = RBF(10, (1e-2, 1e2))
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=4)

gp_model.fit(X_train, y_train)
y_test_pred, sigma = gp_model.predict(X_test, return_std=True)
y_train_pred, sigma = gp_model.predict(X_train, return_std=True)
#%% 
 
plt.figure()
plt.scatter(y_train, y_train_pred,c="b", marker="o", label="training")
plt.scatter(y_test, y_test_pred,c="r", marker="o", label="validation")
plt.xlabel("observed")
plt.ylabel("predicted")
plt.title("GP")
plt.legend()
plt.show()

#%%


df.to_csv('data'+'_n'+str(n)+'.csv',index=False)
#%%
u_old=Function(V1)    
u_new=Function(V0) 
for i in range(10):
    flux0=get_flux_fromGPy(u_old,u_new,gp_model)
    u1=Nsolve(submesh1, flux0,f)
    u_old = u_new
    u_new= u1


 