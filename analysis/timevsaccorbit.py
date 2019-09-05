#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:04:28 2019

@author: bert
"""

import pickle
import pykep as pk
import scipy as sp

from matplotlib import pyplot as plt
from time import time

# Load harmonics model
print("Loading gravity model...", end="")
radius, mu, c, s, max_degree, max_order = \
    pk.util.load_gravity_model("/home/bert/miniconda3/envs/pykep/lib/python3.7/site-packages/pykep/util/gravity_models/Moon/grgm_1200a_t.txt")
    
print(", loaded")    


print("Loading benchmark...", end="")
with open("benchmarkorbit390.pkl", "rb") as f:
    benchmark = pickle.load(f)
    
print(", loaded")
    
# Define coordinates
print("Defining states")
e = 0.05
a = (radius + 390e03)/(1 + e)
i = sp.radians(92)
raan_list = sp.arange(0, sp.pi, sp.pi/10)
aop = 3*sp.pi/2
ta = 0

period = 2*sp.pi * sp.sqrt(a**3/mu)

# Function to calculate the state derivative.
def propagate(t, state):
    # Rotational rate of the Moon
    w_m = 2.661695728e-6  # rad/s

    # Get the absolute rotation (assuming alpha(t0) = 0)
    alpha = w_m * t

    # Build the rotation matrix
    sin = sp.sin(alpha)
    cos = sp.cos(alpha)
    
    transformation = sp.array([[cos, sin, 0],
                               [-sin, cos, 0],
                               [0, 0, 1]])
    
    full_transformation = sp.zeros(2 * [state.shape[0]//2])
        
    for i in range(0, full_transformation.shape[0], 3):
        full_transformation[i:i+3, i:i+3] = transformation
        
    # Get the state in the inertial reference frame
    dim_state = sp.reshape(state, (state.shape[0]//6, 6))
    
    rotated_pos = sp.dot(full_transformation, dim_state[:, 0:3].flatten())
    
    dim_rotated_pos = sp.reshape(rotated_pos, (rotated_pos.shape[0]//3, 3))

    # gravity_spherical_harmonic needs an (N x 3) input array
    acc = pk.util.gravity_spherical_harmonic(dim_rotated_pos, radius, mu, c, s, degree, order)
    
    delta_state = sp.column_stack((dim_state[:, 3],
                                   dim_state[:, 4],
                                   dim_state[:, 5],
                                   acc[:, 0],
                                   acc[:, 1],
                                   acc[:, 2])).flatten()

    return delta_state

states = []
for raan in raan_list:
    states.append(sp.concatenate(pk.core.par2ic((a, e, i, raan, aop, ta), mu)))
    
state0 = sp.concatenate(states)

print("propagating")
for degree in range(0, 40, 5):
    for order in [0, degree]:
        print(f"\rNow calculating with degree and order {degree}/{order}\t", end="")
        
        t1 = time()
        solution = sp.integrate.solve_ivp(propagate, (0, period), state0, rtol=1e-12, atol=1e-12)
        t2 = time()

        state_hist = sp.transpose(solution['y'])
        
        plt.figure(1, figsize=(12, 9))
        
        accuracy = sp.linalg.norm(sp.reshape(state_hist[-1] - benchmark.flatten(), (benchmark.shape[0], 6))[:, 0:3], axis=1)
        
        plt.plot(sp.average(accuracy), t2-t1, ".")
        plt.text(sp.average(accuracy), t2-t1, f"{degree}/{order}", fontsize=8)
        
plt.xlabel("Accuracy [m]")
plt.ylabel("Run time [s]")

plt.savefig("timevsacc390orbit")
