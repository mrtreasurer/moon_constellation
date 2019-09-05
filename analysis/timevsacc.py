#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:43:56 2019

@author: bert
"""

import numpy as np
import pickle as pkl
import pykep as pk

from matplotlib import pyplot as plt
from time import time


def get_stats(function, args):
    t_list = []
    
    for k in range(10):
        t1 = time()
        acc = function(*args)
        t2 = time()
        
        t_list.append(t2 - t1)
        
    t_elapsed = np.average(t_list)
    
    return t_elapsed, acc


# Load harmonics model
print("Loading gravity model")
radius, mu, c, s, max_degree, max_order = \
    pk.util.load_gravity_model("/home/bert/miniconda3/envs/pykep/lib/python3.7/site-packages/pykep/util/gravity_models/Moon/grgm_1200a_t.txt")
    
# load benchmark
print("Loading benchmark")
with open("benchmark50.pkl", "rb") as f:
    benchmark = pkl.load(f)
    
# Define coordinates
print("Defining state")
e = 0.05
a = (radius + 50e03)/(1 - e)
i = np.radians(92)
raan = 0
aop = 3*np.pi/2
ta = np.pi/6

state = np.array([pk.core.par2ic((a, e, i, raan, aop, ta), mu)[0]])

# Dummy run
print("Performing dummy run")
pk.util.gravity_spherical_harmonic(state, radius, mu, c, s, 1, 1)

# Calculate accuracies and time
print("Looping")
plt.figure(1, figsize=(24, 18))

times = []
accuracies = []

n_list = np.arange(0, 100, 5, dtype=int)
m_list = np.arange(0, 100, 5, dtype=int)

for n in n_list:
    for m in [0, n]:
        print(f"\rNow calculating with degree and order {n}/{m}\t", end="")
        t, acc = get_stats(pk.util.gravity_spherical_harmonic, [state, radius, mu, c, s, n, m])
    
        times.append(t)
        
        prec = np.linalg.norm(benchmark-acc)
        accuracies.append(prec)
        
        plt.plot(prec, t, 'bo')
        plt.text(prec + 0.00001, t, f"{n}/{m}", fontsize=8)
    
plt.xlabel("accuracy [m/s2]")
plt.ylabel("run time [s]")

plt.contour(accuracies, times, np.column_stack(len(times) * np.array(accuracies)) + np.row_stack(len(accuracies) * np.array(times)))

plt.savefig("plots/timevsacc390")    
    