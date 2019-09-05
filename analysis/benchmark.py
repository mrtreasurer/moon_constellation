#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:16:20 2019

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
a = (radius + 50e03)/(1 - e)
i = np.radians(92)
raan = 0
aop = 3*np.pi/2
ta = np.pi/6

state = np.array([pk.core.par2ic((a, e, i, raan, aop, ta), mu)[0]])

print("Calculating benchmark")
benchmark = pk.util.gravity_spherical_harmonic(state, radius, mu, c, s, max_degree, max_order)[0]

with open("benchmark50.pkl", "wb") as f:
    pickle.dump(benchmark, f)