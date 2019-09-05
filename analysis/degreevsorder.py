#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:20:19 2019

@author: bert
"""

import numpy as np
import pykep as pk

from matplotlib import pyplot as plt
from time import time


def get_time(function, args):
    t_list = []
    
    for k in range(10):
        t1 = time()
        dummy = function(*args)
        t2 = time()
        
        t_list.append(t2 - t1)
        
    t_elapsed = np.average(t_list)
    
    return t_elapsed


# Load harmonics model
print("Loading gravity model")
radius, mu, c, s, max_degree, max_order = \
    pk.util.load_gravity_model("/home/bert/miniconda3/envs/pykep/lib/python3.7/site-packages/pykep/util/gravity_models/Moon/grgm_1200a_t.txt")
    
# Define coordinates
print("Defining state")
e = 0.05
a = (radius + 50)/(1 - e)
i = np.radians(92)
raan = 0
aop = 270
ta = np.pi/6

state = np.array([pk.core.par2ic((a, e, i, raan, aop, ta), mu)[0]])

# Dummy run for jit
print("Preforming dummy run")
pk.util.gravity_spherical_harmonic(state, radius, mu, c, s, 1, 1)

# Plot square harmonics
print("Looping over degree and order")
n_range = np.arange(1, 21, 1, dtype=int)
m_range = np.arange(1, 11, 1, dtype=int)

t = np.zeros((20, 10))

for i, n in enumerate(n_range):
    print(f"\r{i+1}\t", end="")
    for j, m in enumerate(m_range[:n]):
        t[i, j] = get_time(pk.util.gravity_spherical_harmonic, [state, radius, mu, c, s, n, m])
        
# Plotting
ax = plt.figure(1, figsize=(12, 9))
cs = plt.contour(m_range, n_range, t)
plt.xlabel("order")
plt.ylabel("degree")
plt.clabel(cs, inline=1, fontsize=10)
    
plt.savefig("plots/degreevsorder")