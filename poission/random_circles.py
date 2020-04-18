#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:15:08 2020

@author: 212731466
"""
import numpy as np
import matplotlib.pyplot as plt
import time
timeout = time.time() + 60*1   # 1 minutes from now

nCircles = 20
circles = np.zeros((nCircles ,2));
r = 0.05;
generated_cicles=[]
distFromPrevCircles=np.zeros((nCircles ,1));
fig, ax = plt.subplots(1)
#%%
np.random.seed(24)
a=2*r
b=1.0-2*r
for i in range(nCircles):
    newCircleFound = True
    #loop iteration which runs until finding a circle which doesnt intersect with previous ones
    while newCircleFound:
        x = a + (b -a)*np.random.rand(1)
        y = a + (b -a)*np.random.rand(1)
        if  np.abs(x-0.5)> 1.5*r:
        #calculates distances from previous drawn circles
            prevCirclesX = circles[0:i,0]        
            prevCirclesY = circles[0:i,1];
            distFromPrevCircles=((prevCirclesX-x)**2+(prevCirclesY-y)**2)**0.5
    
            #if the distance is not to small - adds the new circle to the list
            if all(distFromPrevCircles>3*r):
                newCircleFound = False;
                circles[i,:] = [x ,y];
                print(x,y,r)
                generated_cicles.append([(x[0],y[0]),r])
                circle1=plt.Circle((x[0], y[0]), r, color='r')
                ax.add_artist(circle1)
            if time.time() > timeout:
                break
       
#plt.xlim([0,nCircles+1])
#plt.ylim([0,nCircles+1])
plt.show()            

import pandas as pd
df = pd.DataFrame()
df['x']=circles[:,0]
df['y']=circles[:,1]

df.to_csv('mesh_with_holls_05.csv',index=False)