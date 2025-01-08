import os

import matplotlib.pyplot as plt
import pandas as pd


def tensorboard_smoothing(x, smooth=0.6):
    x = x.copy()
    weight = smooth
    for i in range(1, len(x)):
        x[i] = (x[i - 1] * weight + x[i]) / (1 + weight)
        weight = (weight + 1) * smooth
    return x


fig, ax1 = plt.subplots(1, 1)  # a figure with a 2x1 grid of Axes
len_mean_single = pd.read_csv("quantizer_12_medium_b[1].csv")
len_mean_batch = pd.read_csv("quantizer_12_medium_b[8].csv")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1.plot(
    len_mean_single["Step"],
    tensorboard_smoothing(len_mean_single["Value"], smooth=0.75),
    color="#3399FF",
)

ax1.plot(
    len_mean_batch["Step"],
    tensorboard_smoothing(len_mean_batch["Value"], smooth=0.75),
    color="#FF6666",
)

ax1.legend(["batch_size = 1", "batch_size = 8"], loc="upper right")

# ax1.set_xticks(np.arange(0, 24, step=2))
ax1.set_xlabel("timesteps")
ax1.set_ylabel("NSM(steps)", color="black")

plt.show()

os.makedirs("./figures", exist_ok=True)

fig.savefig(fname="./figures/ep_len_mean" + ".pdf", format="pdf")
