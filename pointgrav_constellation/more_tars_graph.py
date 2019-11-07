import numpy as np

from matplotlib import pyplot as plt

from constants import _target_coors_deg as target_coors


def init_graph():
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)

    plt.xticks(np.linspace(-180, 180, 13))
    plt.yticks(np.linspace(-90, 90, 13))

    plt.plot(target_coors[:, 1] - (target_coors[:, 1] > 180)*360, target_coors[:, 0], 'kx')


results = np.genfromtxt("data/more_tars_results.csv", delimiter=",")

path = "data/tars/targets_{extra}_{iteration}.csv"

extra_tars = 12
iterations = 100

colors = ["b", "r", "g"]

extra = 12

plt.figure(1)

init_graph()

plt.figure(2)

init_graph()

for iteration in range(iterations):
    coors = np.genfromtxt(path.format(extra=extra, iteration=iteration), delimiter=",")

    success, _extra, _iteration, mean_charge, min_charge = results[iterations*(extra - 1) + iteration]

    if success:
        plt.figure(1)

    else:
        plt.figure(2)

    if extra == 1:
        plt.plot((coors[1] - (coors[1] > np.pi)*2*np.pi)*180/np.pi, coors[0]*180/np.pi, "bo")

    else:
        plt.plot((coors[:, 1] - (coors[:, 1] > np.pi)*2*np.pi)*180/np.pi, coors[:, 0]*180/np.pi, "o")
        # for coor in coors:
            # plt.text((coor[1] - (coor[1] > np.pi)*2*np.pi)*180/np.pi, coor[0]*180/np.pi, f"{iteration + 1}")
            

# plt.show()

for extra in range(1, extra_tars + 1):
    rge = results[iterations*(extra-1):iterations*extra]

    if len(rge[rge[:, 0] == 1]) != 0:
        print(f"{extra}:\tsuccess rate: {np.count_nonzero(rge[:, 0])}%\t" + 
                f"mean: {np.round(np.mean(rge[:, 3][rge[:, 0] == 1])/5, 3)}\tmin: {np.round(np.mean(rge[:, 4][rge[:, 0] == 1])/5, 3)}")