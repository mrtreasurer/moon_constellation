#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:18:12 2019

@author: bert
"""

import numpy as np
import pykep as pk

from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import constants as cte

from definitions import d2_gamma, d_jac, jacobian
from propagation import dx_dt, rotate
from d2pot import d2u_dx2
from potential import potential


x0 = -1.311519120505e-2
y0 = 5.435394815081e-4
z0 = 0.

u0 = -1.281711107594e-2
v0 = -3.056086111584e-1
w0 = 9.077797920947e-1

state0 = np.array([x0, y0, z0, u0, v0, w0])

a, e, i, raan, aop, ea = pk.core.ic2par(state0[0:3], state0[3:6], cte.mu_n)
period = 2*np.pi * np.sqrt(a**3/cte.mu_n)

end_time = 2*np.pi

def terminate(t, state):
    if t >= end_time:
        return state[2]

    else:
        return 1


eps = 1e-6
dist = 1e-11

u0 = potential(state0[0:3], cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)
c0 = jacobian(state0, cte.mu_n, u0)

iteration = 0

converged = False
while not converged:
    print(f"Iteration: {iteration}")

    terminate.terminal = True
    terminate.direction = state0[5]

    solution = solve_ivp(dx_dt, (0, 1.2*end_time), state0, events=terminate, rtol=1e-10, atol=1e-10)
    print(solution.message)

    state_hist = np.transpose(solution.y)
    time_hist = solution.t

    r_state_hist = rotate(time_hist, state_hist)

    state_t = r_state_hist[-1]
    
#    plt.figure(1)
#    ax = plt.axes(projection='3d')
#    ax.plot(r_state_hist[:, 0], r_state_hist[:, 1], r_state_hist[:, 2])
#    ax.plot([0], [0], [0], "ko")
#    ax.plot([state0[0]], [state0[1]], [state0[2]], "o")
#    ax.plot([state_t[0]], [state_t[1]], [state_t[2]], "o")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    ax.set_zlabel("z")
    
    ut = potential(state_t[0:3], cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)
    c_t = jacobian(state_t, cte.mu_n, ut)

    req = np.linalg.norm(state_t[0:3] - state0[0:3])/np.linalg.norm(state0[0:3]) + \
        np.linalg.norm(state_t[3:6] - state0[3:6])/np.linalg.norm(state0[3:6]) + \
        abs((c_t - c0)/c0)
    print(f"Miss distance: {req}")

    if req < dist:
        converged = True
        print(f"state0: {state0}")

    else:
        k = np.append(np.delete(state_t-state0, 2), c_t - c0)

        dtime = time_hist[1::] - time_hist[:-1:]
        
        print("State transition")
        
        phi = np.eye(6)
        for state, dt in zip(r_state_hist[1::], dtime):
            df_dx = np.zeros((6, 6))

            df_dx[0, 3] = 1
            df_dx[1, 4] = 1
            df_dx[2, 5] = 1
            df_dx[3, 3] = 2
            df_dx[4, 4] = -2

            d2u = d2u_dx2(state[0:3], cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)
            d2g = d2_gamma(state[0:3], cte.mu_n, d2u)

            df_dx[3, 0:3] = -d2g[0]
            df_dx[4:6, 0:3] = d2g[1:3]

            phidot = np.dot(df_dx, phi)
            phi += phidot * dt
            
#    break

        det = np.delete(dx_dt(time_hist[-1], state_t), 2)

        det_de0 = np.delete(np.delete(phi, 2, 0), 2, 1) - 1/state_t[5]*det*np.delete(phi[2], 2)

        print("svd")
        
        dk = np.zeros((6, 5))
        dk[0:5, 0:5] = det_de0 - np.eye(5)

        acc = pk.util.gravity_spherical_harmonic(np.array([state0[0:3]]), cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)
        dc = d_jac(state0, cte.mu_n, acc[0])

        dk[5] = np.delete(dc, 2)

        u, d, vt = np.linalg.svd(dk)
        
        print("State update")

        flat_s = np.zeros(d.shape)
        for i, a in enumerate(d):
            if a > eps:
                flat_s[i] = 1/a

            else:
                flat_s[i] = 0

        s = np.column_stack((np.diag(flat_s), np.zeros(5)))

        v = np.transpose(vt)
        delta_state0 = np.insert(-np.dot(np.linalg.multi_dot([v, s, np.transpose(u)]), k), 2, 0)

        state0 += delta_state0
        
        iteration += 1
