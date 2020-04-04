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
import pandas as pd
import GPy

#%% get interface values working with vectors
def get_interface_values_from_vector(V0,mesh0,u0_vec):
    InterfaceBoundary = AutoSubDomain(lambda x, on_bnd: near(x[0], 1.0) and on_bnd)
    bc_Iinterface = DirichletBC(V0, 1, InterfaceBoundary)

    interface_indecater = Function(V0)
    bc_Iinterface.apply(interface_indecater.vector())
    index0=interface_indecater.vector().get_local().astype(bool)
    # interface u
    ub=u0_vec.get_local()[index0]
    X0=V0.tabulate_dof_coordinates()[index0,:]

    return ub,X0,index0


#%% # get interface values working with functions
def get_interface_values_from_function(V0,mesh0,u0_fun):
    InterfaceBoundary = AutoSubDomain(lambda x, on_bnd: near(x[0], 1.0) and on_bnd)
    bc_Iinterface = DirichletBC(V0, 1, InterfaceBoundary)

    interface_indecater = Function(V0)
    bc_Iinterface.apply(interface_indecater.vector())
    index0=interface_indecater.compute_vertex_values(mesh0).astype(bool)
    # interface u
    ub=u0_fun.compute_vertex_values(mesh0)[index0]
    X0=mesh0.coordinates()[index0,:]

    return ub,X0,index0

#%%

#%% solve subdomain 0
mesh0 = RectangleMesh(Point(0.0, 0.0), Point(1.0,1.0), 30, 30)
# set Dirichlet BC value
initial=0.1
left=0.0
top=1.0
right=1.0
bottom=0.0

V0 = FunctionSpace(mesh0, "CG", 1)
v2d=vertex_to_dof_map(V0)
d2v=dof_to_vertex_map(V0)

# create an intial vector
u0= Function(V0)
b=u0.vector()

ub0, X0, b_vector_index=get_interface_values_from_vector(V0,mesh0,b)
b_array=np.zeros(mesh0.num_vertices())

df=pd.DataFrame()
df_k=pd.DataFrame()
for isample in np.random.random(10):

    #b_array[b_vector_index]=np.random.random((b_vector_index.sum().astype(int)))*0+isample
    b_array[b_vector_index]=isample
    b.set_local(b_array)
    u1 = Vector2Function(V0,b)

    ub = b[b_vector_index]
    u0=Dirichlet(mesh0,0.0,left,top,right,bottom,True,u1)
    lambda0 = GetFlux(mesh0,u0)


    lamb,X0,index0=get_interface_values_from_vector(V0,mesh0,lambda0)

    df_k['x']=X0[:,1]
    df_k['ub']=isample
    df_k['lambda']=lamb

    df=df.append(df_k,ignore_index=True)

#%%

kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
X = np.zeros((df.shape[0],2))
X[:,0] = df['x'].values
X[:,1] = df['ub'].values
Y = df['lambda'].values.reshape(df.shape[0],1)

gp_model = GPy.models.GPRegression(X,Y,kernel)
gp_model.optimize(messages=True)
gp_model.plot()
#%%
#I am here !
mesh1 = RectangleMesh(Point(1.0, 0), Point(2,1.0), 30, 30)


V1 = FunctionSpace(mesh1, "CG", 1)



u1= Function(V1)
lambda1=u1.vector()

ub1, X1, lambda1_vector_index=get_interface_values_from_vector(V1,mesh1,lambda1)
lambda1_array=np.zeros(mesh1.num_vertices())

ub_sample=0.1

num_boundary=X1.shape[0]
Xnew=np.array([X1[:,1], ub_sample*np.ones(num_boundary)]).T.reshape(-1,2)

predicted_flux=gp_model.predict(Xnew)
predicted_flux=predicted_flux[0]

lambda1_array[lambda1_vector_index]=predicted_flux.reshape(-1,)
lambda1.set_local(lambda1_array)


u1=Neumann(mesh1,lambda1)