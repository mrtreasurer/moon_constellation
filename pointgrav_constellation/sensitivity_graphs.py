import numpy as np

from matplotlib import pyplot as plt

import constants as cte

save = True


inc_data = np.genfromtxt("data/inc_sensitivity.csv", delimiter=",")
sma_data = np.genfromtxt("data/sma_sensitivity.csv", delimiter=",")

inc_data = inc_data[inc_data[:, 2] != 0]
sma_data = sma_data[sma_data[:, 2] != 0]

plt.figure(1, dpi=300)
plt.title("Constellation Sensitivity to Variation in Inclination")
plt.xlabel("Inclination [deg]")
plt.ylabel("Battery charge [%]")

plt.plot(2*[cte.optim_inc*180/np.pi], [0, 100], "k", linewidth=1)

plt.plot(inc_data[:, 0]*180/np.pi, inc_data[:, 1]/cte.tar_battery_cap*100, c="#285FAC", label="Mean Charge")
plt.plot(inc_data[:, 0]*180/np.pi, inc_data[:, 2]/cte.tar_battery_cap*100, c="#285FAC", linestyle="--", label="Minimum Charge")

plt.legend(loc="lower left")

if save:
    plt.savefig("data/inc_sensitivity")

plt.figure(2, dpi=300)
plt.title("Constellation Sensitivity to Variaiton in Semi-Major Axis")
plt.xlabel("Semi-Major Axis [km]")
plt.ylabel("Battery charge [%]")

plt.plot(2*[cte.optim_sma/1e3], [0, 100], "k", linewidth=1)

plt.plot(sma_data[:, 0]/1e3, sma_data[:, 1]/cte.tar_battery_cap*100, c="#285FAC", label="Mean Charge")
plt.plot(sma_data[:, 0]/1e3, sma_data[:, 2]/cte.tar_battery_cap*100, c="#285FAC", linestyle="--", label="Minimum Charge")

plt.legend()

if save:
    plt.savefig("data/sma_sensitivity")


if not save:
    plt.show()