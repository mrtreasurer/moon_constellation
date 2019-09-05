#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:08:36 2019

@author: bert
"""

import numpy as np
import pykep as pk

import constants as c

from definitions import distances


def rotate(theta, state):
    def rot(th, s):
        rotation = np.array([[np.cos(th), -np.sin(th), 0.],
                             [np.sin(th), np.cos(th), 0.],
                             [0., 0., 1.]])

        new_pos = np.dot(rotation, s[0:3])
        new_vel = np.dot(rotation, s[3:6] + np.cross(np.array([0, 0, 1]), s[0:3]))

        return np.concatenate((new_pos, new_vel))

    if state.ndim == 1:
        return rot(theta, state)

    elif state.ndim == 2:
        new_state = np.zeros(state.shape)

        for i in range(len(state)):
            new_state[i] = rot(theta[i], state[i])

        return new_state


def dx_dt(t, state):

    x = state[0]
    y = state[1]
    z = state[2]
    u = state[3]
    v = state[4]
    w = state[5]

    acc = pk.util.gravity_spherical_harmonic(np.array([state[0:3]]), c.r_n, c.mu_n, c.cos, c.sin, c.degree, c.order)[0]

    r, rho = distances(state[0:3])

    dg_dx = (x - 1 + c.mu_n) - (1 - c.mu_n) * (x - 1) / rho**3 + acc[0]
    dg_dy = y - (1 - c.mu_n) * y / rho**3 + acc[1]
    dg_dz = - (1 - c.mu_n) * z / rho**3 + acc[2]

    dx = np.array([u, v, w, 2*v + dg_dx, -2*u + dg_dy, dg_dz])

    return dx
