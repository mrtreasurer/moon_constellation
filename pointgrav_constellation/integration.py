# -*- coding: utf-8 -*-

import numpy as np

#from numba import jit


#@jit
def rk4(t, x0, f):
    state = np.zeros((len(t), len(x0)))
    state[0] = x0
    
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        
        a = dt * f(t[i], state[i])
        b = dt * f(t[i] + dt/2, state[i] + a/2)
        c = dt * f(t[i] + dt/2, state[i] + b/2)
        d = dt * f(t[i] + dt, state[i] + c)
        
        state[i+1] = state[i] + 1/6 * (a + 2*b + 2*c + d)
        
    return state


if __name__ == "__main__":
#    @jit
    def function(x, y):
        return x * np.sqrt(y)
    
    time = np.arange(0, 10.1, 0.1)
    
    result = rk4(time, [1], function)
    
    for x, y in list(zip(time, result))[::10]:
        print("%4.1f %10.5f %+12.4e" % (x, y, y - (4 + x**2)**2 / 16))
        