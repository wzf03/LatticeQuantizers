from pathlib import Path

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
default_single = pd.read_csv(
    "./data/compare_batch[1]_batch[8]/logs/quantizer_12_medium_v[1000]_T[1000000]_b[1].csv"
)
default_batch = pd.read_csv(
    "./data/compare_batch[1]_batch[8]/logs/quantizer_12_medium_v[1000]_T[1000000]_b[8].csv"
)
cosine_batch = pd.read_csv(
    "./data/compare_batch[1]_batch[8]/logs/quantizer_n12_T1000000_B8.csv"
)


ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1.plot(
    default_single["Step"],
    tensorboard_smoothing(default_single["Value"], smooth=0.75),
    color="#3399FF",
)

ax1.plot(
    default_batch["Step"],
    tensorboard_smoothing(default_batch["Value"], smooth=0.75),
    color="#FF6666",
)

ax1.plot(
    cosine_batch["Step"],
    tensorboard_smoothing(cosine_batch["Value"], smooth=0.75),
    color="gray",
)

ax1.legend(
    ["medium, batch_size = 1", "medium, batch_size = 8", "cosine, batch_size = 8"],
    loc="upper right",
)

ax1.set_xlabel("timesteps")
ax1.set_ylabel("NSM(steps)", color="black")

plt.show()

figures_path = Path("./figures")
figures_path.mkdir(parents=True, exist_ok=True)

fig.savefig(fname=figures_path / "compare.pdf", format="pdf")
