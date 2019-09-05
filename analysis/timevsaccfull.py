#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:21:49 2019

@author: bert
"""
import numpy as np
import pickle as pkl
import pykep as pk

from matplotlib import pyplot as plt
from time import time


def get_stats(function, args):
    t_list = []
    
    for k in range(1):
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
with open("benchmark390full.pkl", "rb") as f:
    benchmark = pkl.load(f)
    
# Define coordinates
print("Defining state")
e = 0.05
a = (radius + 390e03)/(1 + e)
i = np.radians(92)
raan_list = np.arange(0, np.pi, np.pi/10)
aop = 3*np.pi/2
ta_list = np.arange(0, 2*np.pi, 2*np.pi/10)


n_list = np.arange(0, 40, 5, dtype=int)
m_list = np.arange(0, 40, 5, dtype=int)

plt.figure(1, figsize=(24, 18))

for n in n_list:
    
    for m in [0, n]:
        
        n_times = []
        n_accuracies = []
        
        for j, raan in enumerate(raan_list):
            
            for k, ta in enumerate(ta_list):
                print(f"\rNow calculating at raan = {raan} and ta = {ta} with degree and order {n}/{m}\t", end="")
                
                state = np.array([pk.core.par2ic((a, e, i, raan, aop, ta), mu)[0]])
        
                if n == 0 and m == 0 and j == 0 and k == 0:
                    # Dummy run
                    print("Performing dummy run")
                    pk.util.gravity_spherical_harmonic(state, radius, mu, c, s, 1, 1)

                t, acc = get_stats(pk.util.gravity_spherical_harmonic, [state, radius, mu, c, s, n, m])
            
                n_times.append(t)
                
                prec = np.linalg.norm(benchmark[j, k] - acc)
                n_accuracies.append(prec)
                
        t = np.average(n_times)
        prec = np.average(n_accuracies)
                
        plt.plot(prec, t, '.')
        plt.text(prec, t, f"{n}/{m}", fontsize=8)
    
plt.xlabel("accuracy [m/s2]")
plt.ylabel("run time [s]")

plt.savefig("plots/timevsacc390full")    
    