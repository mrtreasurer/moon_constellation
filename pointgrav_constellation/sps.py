# -*- coding: utf-8 -*-
import numpy as np
import pykep as pk

#from scipy.integrate import solve_ivp

#from propagation import dx_dt
    

def create_sats(kep0, mu_m, sim_time):
    sats = np.zeros((len(sim_time), len(kep0), 3))
    for i, k0 in enumerate(kep0):
        
        rotation = False
        index = -1
        for j, ki in enumerate(kep0[:i]):
            if np.all(np.delete(k0, 3) == np.delete(ki, 3)):
                rotation = True
                index = j
        
        if rotation:
            raan = k0[3] - kep0[index][3]
            rot = np.array([[np.cos(raan), -np.sin(raan), 0],
                            [np.sin(raan), np.cos(raan), 0],
                            [0, 0, 1]])
    
            pos = np.transpose(np.dot(rot, np.transpose(sats[:, index])))
            
        else:
            mean_motion = np.sqrt(mu_m/k0[0]**3)
            ta = (k0[5] + mean_motion * sim_time) % (2*np.pi)
            
            k = np.empty((sim_time.shape[0], 6))
            for j in range(k0.shape[0] - 1):
                k[:, j] = k0[j]
            k[:, 5] = ta
            
            pos = np.empty((k.shape[0], 3))
            for j in range(k.shape[0]):
                pos[j] = pk.core.par2ic(k[j], mu_m)[0]
            
#            cart0 = np.concatenate(pk.core.par2ic(k0, mu_m))                
#            pos = np.transpose(solve_ivp(dx_dt, (sim_time[0], sim_time[-1]), cart0, method='LSODA', t_eval=sim_time, rtol=1e-10, atol=1e-10).y[0:3])
        
        sats[:, i] = pos
        
    return sats


#if __name__ == "__main__":
#    print(mu_sh, r_sh)
#    
#    print(np.linalg.norm(sats_pos[0, 3:6]))
#    print(np.sqrt(mu_sh/sma))
