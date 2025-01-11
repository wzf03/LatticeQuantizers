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


fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(8, 4))  # a figure with a 2x1 grid of Axes

# Data for dimension 16
default_single_16 = pd.read_csv(
    "./data/compare_batch[1]_batch[8]_d16/logs/quantizer_medium_n16_T1000000_B1_2025-01-11_15_48_21.csv"
)
default_batch_16 = pd.read_csv(
    "./data/compare_batch[1]_batch[8]_d16/logs/quantizer_medium_n16_T1000000_B8_2025-01-11_15_48_38.csv"
)
cosine_batch_16 = pd.read_csv(
    "./data/compare_batch[1]_batch[8]_d16/logs/quantizer_cosine_n16_T1000000_B8_2025-01-11_15_46_49.csv"
)

# Plot for dimension 16
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1.plot(
    default_single_16["Step"],
    tensorboard_smoothing(default_single_16["Value"], smooth=0.75),
    color="#3399FF",
)

ax1.plot(
    default_batch_16["Step"],
    tensorboard_smoothing(default_batch_16["Value"], smooth=0.75),
    color="#FF6666",
)

ax1.plot(
    cosine_batch_16["Step"],
    tensorboard_smoothing(cosine_batch_16["Value"], smooth=0.75),
    color="gray",
)

ax1.legend(
    ["medium, batch_size = 1", "medium, batch_size = 8", "cosine, batch_size = 8"],
    loc="upper right",
)

ax1.set_title("dimension-16")
ax1.set_xlabel("timesteps")
ax1.set_ylabel("NSM(steps)", color="black")

# Data for dimension 12
default_single_12 = pd.read_csv(
    "./data/compare_batch[1]_batch[8]_d12/logs/quantizer_12_medium_v[1000]_T[1000000]_b[1].csv"
)
default_batch_12 = pd.read_csv(
    "./data/compare_batch[1]_batch[8]_d12/logs/quantizer_12_medium_v[1000]_T[1000000]_b[8].csv"
)
cosine_batch_12 = pd.read_csv(
    "./data/compare_batch[1]_batch[8]_d12/logs/quantizer_n12_T1000000_B8.csv"
)

# Plot for dimension 12
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2.plot(
    default_single_12["Step"],
    tensorboard_smoothing(default_single_12["Value"], smooth=0.75),
    color="#3399FF",
)

ax2.plot(
    default_batch_12["Step"],
    tensorboard_smoothing(default_batch_12["Value"], smooth=0.75),
    color="#FF6666",
)

ax2.plot(
    cosine_batch_12["Step"],
    tensorboard_smoothing(cosine_batch_12["Value"], smooth=0.75),
    color="gray",
)

ax2.legend(
    ["medium, batch_size = 1", "medium, batch_size = 8", "cosine, batch_size = 8"],
    loc="upper right",
)

ax2.set_title("dimension-12")
ax2.set_xlabel("timesteps")
ax2.set_ylabel("NSM(steps)", color="black")

plt.tight_layout()

figures_path = Path("./figures")
figures_path.mkdir(parents=True, exist_ok=True)

fig.savefig(fname=figures_path / "compare_12_16.pdf", format="pdf")
plt.show()
