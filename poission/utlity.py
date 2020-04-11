#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:52:02 2020

@author: 212731466
"""
from dolfin import *
import numpy as np

def GetFlux(mesh,u):
    # Create subdomain (x0 = 1)

    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)

    class Interface(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1)


    interfacet = Interface()

    # Mark facets
    interfacet_boundary = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    interfacet_boundary.set_all(0)
    interfacet.mark(interfacet_boundary, 1)
    ds = Measure("ds")[interfacet_boundary]

    ### First method ###
    # Define facet normal vector (built-in method)
    n = FacetNormal(mesh)
    m1 = dot(grad(u), n)*v*ds(1)
    flux_vector = assemble(m1)
    
    flux_func=Vector2Function(V,flux_vector)

    return flux_vector,flux_func

def Vector2Function(V,b):
     u= Function(V)
     if isinstance(b,Vector):
         values=b.get_local()
     else:
         values=b
     u.vector().set_local(values)
     return(u)