#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:56:13 2019

@author: bert
"""

import numpy as np
import pickle
import pykep as pk


# Load harmonics model
print("Loading gravity model")
radius, mu, c, s, max_degree, max_order = \
    pk.util.load_gravity_model("/home/bert/miniconda3/envs/pykep/lib/python3.7/site-packages/pykep/util/gravity_models/Moon/grgm_1200a_t.txt")
    
# Define coordinates
print("Defining state")
e = 0.05
a = (radius + 390e03)/(1 + e)
i = np.radians(92)
raan_list = np.arange(0, np.pi, np.pi/10)
aop = 3*np.pi/2
ta_list = np.arange(0, 2*np.pi, 2*np.pi/10)

benchmark = np.zeros((10, 10, 3))

for j, raan in enumerate(raan_list):
    for k, ta in enumerate(ta_list):
        state = np.array([pk.core.par2ic((a, e, i, raan, aop, ta), mu)[0]])

        print(f"\rCalculating benchmark for raan = {raan}, ta = {ta}\t\t\t\t", end="")
        benchmark[j, k] = pk.util.gravity_spherical_harmonic(state, radius, mu, c, s, max_degree, max_order)[0]

with open("benchmark390full.pkl", "wb") as f:
    pickle.dump(benchmark, f)