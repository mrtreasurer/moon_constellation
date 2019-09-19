# -*- coding: utf-8 -*-

import numpy as np
import pykep as pk

import constants as cte


def dx_dt(t, state):
    r = np.linalg.norm(state[0:3])

    dg = - cte.mu_m/r**3 * state[0:3]

    dx = np.concatenate((state[3:6], dg))

    return dx