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
iterations = 50

results = np.zeros((extra_tars*iterations, 4))

# stats = np.zeros((iterations*extra_tars, 3))

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

        coverage = Coverage(sim_time, sun, targets_pos, cte.r_m, cte.mu_m, cte.dt, cte.min_elev, cte.max_sat_range, cte.ecc, cte.aop, cte.tar_battery_cap, cte.tar_charge_power, cte.sat_las_power, cte.tar_hib_power, cte.sat_point_acc, cte.tar_r_rec, cte.sat_n_las, cte.sat_n_geom, cte.tar_n_rec, cte.sat_wavelength, cte.sat_r_trans)
                
        sats = coverage.create_constellation(sma, inc, n_planes, n_sats_plane)

        # print(coverage.fitness([sma, inc, n_planes, n_sats_plane]))
        charge, dist, eff, contact, targets_in_sunlight, n_targets = coverage.propagate_constellation(sats, sma)

        if np.min(charge) > 0.1*cte.tar_battery_cap:
            np.savetxt(f"data/tars/targets_{extra}_{iteration}.csv", new_coors, delimiter=",")

            # plt.figure(1, dpi=300, clear=True)
            # ax = plt.axes(projection='3d')

            # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # ax.set_xlabel("x [km]")
            # ax.set_ylabel("y [km]")
            # ax.set_zlabel("z [km]")

            # ax.xaxis.set_rotate_label(False)
            # ax.yaxis.set_rotate_label(False)
            # ax.zaxis.set_rotate_label(False)

            # ax.yaxis.set_label_coords(1000, 1000)

            # for i in range(targets_pos.shape[1]):
            #     arr1 = targets_pos[::100, i, 0]/1e3
            #     arr2 = targets_pos[::100, i, 1]/1e3
            #     arr3 = targets_pos[::100, i, 2]/1e3

            #     # np.savetxt(f"data/figure1_targets_{i}.csv", np.column_stack((arr1, arr2, arr3)), delimiter=",")

            #     ax.plot(arr1, arr2, arr3, c='#8262AB', linewidth=1)
            #     ax.plot([arr1[0]], [arr2[0]], [arr3[0]], c="#8262AB", marker='o')

            # for i in range(sats.shape[1]):
            #     arr1 = sats[:, i, 0][sim_time < sat_period]/1e3
            #     arr2 = sats[:, i, 1][sim_time < sat_period]/1e3
            #     arr3 = sats[:, i, 2][sim_time < sat_period]/1e3

            #     # np.savetxt(f"data/figure1_sats_{i}.csv", np.column_stack((arr1, arr2, arr3)), delimiter=",")

            #     ax.plot(arr1, arr2, arr3, c="#285FAC", linewidth=1)
            #     ax.plot([arr1[0]], [arr2[0]], [arr3[0]], c="#285FAC", marker="o")

            # plt.savefig(f"data/tars/figure1_{extra}_{iteration}")
                
            # plt.figure(2, dpi=300, clear=True)
            # for i in range(total_tars):
            #     plt.subplot(total_tars//3 + 1, 3, i + 1)

            #     arr1 = sim_time[np.logical_not(targets_in_sunlight[:, i])]/24/3600
            #     arr2 = charge[:, i][np.logical_not(targets_in_sunlight[:, i])]/cte.tar_battery_cap*100

            #     # np.savetxt(f"data/figure2_{i}.csv", np.column_stack((arr1, arr2)))

            #     plt.plot(arr1, arr2, c="#285FAC")

            #     if i in [3, 4, 5]:
            #         plt.xlabel("Time [days]")
                
            #     if i%3 == 0:
            #         plt.ylabel("Battery charge [%]")

            #     # plt.yticks([80, 85, 90, 95, 100])

            # plt.savefig(f"data/tars/figure2_{extra}_{iteration}")

            # plt.figure(3, dpi=300, clear=True)      
            # arr1 = sim_time/24/3600
            # arr2 = n_targets[:, 0]

            # # np.savetxt("data/figure3.csv", np.column_stack((arr1, arr2)))

            # plt.plot(arr1, arr2, c="#285FAC")

            # plt.xlabel("Time [days]")
            # plt.ylabel("Number of eclipsed targets in view of satellite")

            # plt.savefig(f"data/tars/figure3_{extra}_{iteration}")

            # plt.figure(4, dpi=300, clear=True)
            # for i in range(total_tars):
            #     plt.subplot(total_tars//3 + 1, 3, i + 1)

            #     arr1 = sim_time[np.logical_not(targets_in_sunlight[:, i])]/24/3600
            #     arr2 = np.sum(contact[:, i], axis=1)[np.logical_not(targets_in_sunlight[:, i])]

            #     # np.savetxt(f"data/figure4_{i}.csv", np.column_stack((arr1, arr2)))

            #     plt.plot(arr1, arr2, c="#285FAC")

            #     if i in [3, 4, 5]:
            #         plt.xlabel("Time [days]")
                
            #     if i%3 == 0:
            #         plt.ylabel("Satellites in view during eclipse [-]")

            # plt.savefig(f"data/tars/figure4_{extra}_{iteration}")

            # plt.show()

            results[iterations*(extra - 1) + iteration] = [extra, iteration, np.mean(charge), np.min(charge)]

np.savetxt("data/more_tars_results.csv", results, delimiter=",")
