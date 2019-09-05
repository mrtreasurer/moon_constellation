#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:38:29 2019

@author: bert
"""

import pickle as pkl
import pykep as pk
import scipy as sp

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dopri as dp


# Load harmonics model
print("Loading gravity model")
radius, mu, c, s, max_degree, max_order = \
    pk.util.load_gravity_model("/home/bert/miniconda3/envs/pykep/lib/python3.7/site-packages/pykep/util/gravity_models/Moon/grgm_1200a_t.txt")
    
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

    # Get the state in the inertial reference frame
    rotated_pos = sp.dot(transformation, state[0:3])

    # gravity_spherical_harmonic needs an (N x 3) input array
    acc = pk.util.gravity_spherical_harmonic(sp.array([rotated_pos]), radius, mu, c, s, 10, 10)[0]

    delta_state = sp.array([state[3],
                            state[4],
                            state[5],
                            acc[0],
                            acc[1],
                            acc[2]])

    return delta_state

# Define coordinates
print("Defining state")
e = 0.05
a = (radius + 50)/(1 - e)
i_list = sp.linspace(0, sp.pi, 10)
raan_list = sp.arange(0, sp.pi/2, sp.pi/20)
aop = 270
ta = sp.pi/6

period = 2*sp.pi * sp.sqrt(a**3/mu)

benchmark = {}

t_list = sp.arange(0, 3*period, 10)

for i in i_list:
    for raan in raan_list:
        print(f"\rCalculating benchmark for i = {i}, raan = {raan}\t", end="")
        state0 = sp.concatenate(pk.core.par2ic((a, e, i, raan, aop, ta), mu))
        
        solution = dp.DoPri45integrate(propagate, t_list, state0)
        
        # Calculate benchmark
        benchmark[(i, raan)] = solution
        
        break

    break
        
#with open("benchmark.pkl" , "wb") as f:
#    pkl.dump(benchmark, f)
        
# Transform cartesian to modified equinoctial
eq_state_hist = sp.zeros(solution.shape)
for i, state in enumerate(solution):
    eq_state_hist[i] = pk.ic2par(state[0:3], state[3:6], mu)

labels = [r"$a$", r"$e$", r"$i$", r"$RAAN$", r"$aop$", r"$E$"]
# Plot the orbit
#plt.figure(1, figsize=(16, 3))
#for i in range(6):
#    plt.subplot(1, 6, i + 1)
#    plt.plot(t_list, eq_state_hist[:, i] - eq_state_hist[0, i])
#    plt.xlabel(labels[i])

fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.plot(solution[:, 0], solution[:, 1], solution[:, 2])
