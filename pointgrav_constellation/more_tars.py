import numpy as np

from matplotlib import pyplot as plt

import constants as cte

from optim_class import Coverage, initiate


sma = 2.54279861e6
inc = 1.02596289

n_planes = 4
n_sats_plane = 1

sat_period = 2*np.pi * np.sqrt(sma**3/cte.mu_m)

extra_tars = 12
iterations = 100

results = np.zeros((extra_tars*iterations, 5))

for extra in range(1, extra_tars+1):
    total_tars = cte.target_coors.shape[0] + extra

    for iteration in range(iterations):
        print(f"\r{extra} {iteration}", end="")

        target_coors = cte.target_coors

        new_coors = []
        for i in range(1, extra + 1):
            coor = [cte.r_m,
                    np.pi * np.random.random() - np.pi/2,
                    2*np.pi * np.random.random()
                    ]

            new_coors.append(coor)

        target_coors = np.row_stack((target_coors, new_coors))        

        sim_time, sun, targets_pos = initiate(target_coors, cte.sun_pos0, cte.omega_earth, cte.omega_moon, cte.dt, cte.synodic_period)

        coverage = Coverage(sim_time, sun, targets_pos, cte.r_m, cte.mu_m, cte.dt, cte.min_elev, cte.max_sat_range, cte.ecc, cte.aop, cte.tar_battery_cap, cte.tar_charge_power, 1e5, cte.sat_las_power, cte.tar_hib_power, cte.sat_point_acc, cte.tar_r_rec, cte.sat_n_las, cte.sat_n_geom, cte.tar_n_rec, cte.sat_wavelength, cte.sat_r_trans)
                
        sats = coverage.create_constellation(sma, inc, n_planes, n_sats_plane)

        # print(coverage.fitness([sma, inc, n_planes, n_sats_plane]))
        charge, dist, eff, contact, targets_in_sunlight, n_targets = coverage.propagate_constellation(sats, sma)

        if np.min(charge) > 0.1*cte.tar_battery_cap:
            success = True

        else:
            success = False
        
        np.savetxt(f"data/tars/targets_{extra}_{iteration}.csv", np.array(new_coors)[:,1::], delimiter=",")
        results[iterations*(extra - 1) + iteration] = [success, extra, iteration, np.mean(charge), np.min(charge)]

np.savetxt("data/more_tars_results.csv", results, delimiter=",")
