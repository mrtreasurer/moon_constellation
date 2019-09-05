#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:34:27 2019

@author: bert
"""
import numpy as np
import pickle as pkl
import pykep as pk

from matplotlib import pyplot as plt
from time import time


def get_stats(function, args):
    t_list = []
    
    for k in range(3):
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
with open("benchmark.pkl", "rb") as f:
    benchmark = pkl.load(f)
    
# Define coordinates
print("Defining state")
e = 0.05
a = (radius + 50)/(1 - e)
i_list = np.linspace(0, np.pi, 10)
raan_list = np.arange(0, np.pi/2, np.pi/20)
aop = 270
ta = np.pi/6

accuracies = np.zeros((len(i_list), len(raan_list)))

for j, i in enumerate(i_list):
    for k, raan in enumerate(raan_list):
        print(f"\rCalculating benchmark for i = {i}, raan = {raan}\t", end="")
        state = np.array([pk.core.par2ic((a, e, i, raan, aop, ta), mu)[0]])
        
        t, acc = get_stats(pk.util.gravity_spherical_harmonic, [state, radius, mu, c, s, 35, 35])
    
        accuracies[j, k] = np.linalg.norm(benchmark[f"{i},{raan}"] - acc)

# Plotting
ax = plt.figure(1, figsize=(12, 9))
cs = plt.contour(raan_list, i_list, accuracies)
plt.xlabel("raan")
plt.ylabel("inclination")
plt.clabel(cs, inline=1, fontsize=10)
    
plt.savefig("plots/3535hemisphere")
