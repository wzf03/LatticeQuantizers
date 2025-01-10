import datetime
from pathlib import Path

import numpy as np
from jsonargparse import CLI

from lattice_quantizer.criteria.nsm import nsm_cpu
from lattice_quantizer.lr_scheduler import CosineLR, FactorizedLR, RatioLR  # noqa: F401
from lattice_quantizer.optimizer import SGDLatticeQuantizerOptimizer


def main(
    n: int,
    optimizer: SGDLatticeQuantizerOptimizer,
    output_dir: Path = Path("results"),
    checknsm_num_samples: int = int(1e6),
    checknsm_parallel: bool = False,
):
    output_subdir = (
        output_dir
        / f"quantizer_n{n}_T{optimizer.steps}_B{optimizer.batch_size}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    )
    output_subdir.mkdir(parents=True, exist_ok=True)
    optimizer.log_dir = output_subdir / "logs"

    result = optimizer.optimize(n)
    nsm, var = nsm_cpu(
        result,
        checknsm_num_samples,
        np.random.default_rng(),
        0 if checknsm_parallel else 65536,
    )
    np.savetxt(
        output_subdir / "basis.txt",
        result,
    )
    np.save(output_subdir / "basis.npy", result)

    with (output_subdir / "nsm.txt").open("w") as f:
        nsm_str = (
            f"{np.format_float_positional(nsm)} +/- {np.format_float_scientific(var)}\n"
        )
        f.write(nsm_str)
    print(f"NSM: {nsm_str}")  # noqa: T201


if __name__ == "__main__":
    CLI(main)
