#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:01:06 2019

@author: bert
"""
import pickle
import pykep as pk
import scipy as sp

from matplotlib import pyplot as plt


# Load harmonics model
print("Loading gravity model")
radius, mu, c, s, max_degree, max_order = \
    pk.util.load_gravity_model("/home/bert/miniconda3/envs/pykep/lib/python3.7/site-packages/pykep/util/gravity_models/Moon/grgm_1200a_t.txt")
    
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
    acc = pk.util.gravity_spherical_harmonic(dim_rotated_pos, radius, mu, c, s, 100, 100)
    
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
    
state = sp.concatenate(states)

print("propagating")
solution = sp.integrate.solve_ivp(propagate, (0, period), state, rtol=1e-12, atol=1e-12)

time_hist = solution['t']
state_hist = sp.transpose(solution['y'])

## Transform cartesian to modified equinoctial
#eq_state_hist = sp.zeros(state_hist.shape)
#for i in range(state_hist.shape[0]):
#
#    for j in range(0, state_hist.shape[1], 6):
#        eq_state_hist[i, j:j+6] = pk.ic2par(state_hist[i, j:j+3], state_hist[i, j+3:j+6], mu)
#
#labels = [r"$a \cdot \left( 1 - e^2 \right)$", r"$h$", r"$k$", r"$p$", r"$q$", r"$L$"]
## Plot the orbit
#plt.figure(1, figsize=(16, 3))
#for i in range(6):
#    plt.subplot(1, 6, i + 1)
#    
#    plt.plot(time_hist, eq_state_hist[:, i::6])  # - eq_state_hist[0, i::6])
#        
#    plt.xlabel(labels[i])

benchmark = state_hist[-1]

with open("benchmarkorbit390.pkl", "wb") as f:
    pickle.dump(benchmark, f)
