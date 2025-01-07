import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Check NSM")

    parser.add_argument("--basis", type=str, help="Basis")
    parser.add_argument(
        "--num_samples", type=float, default=1e12, help="Number of samples"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    basis = np.loadtxt(args.basis)
    num_samples = np.int64(args.num_samples)

    from lattice_quantizer.criteria.nsm import nsm_cpu

    nsm, var = nsm_cpu(basis, num_samples, np.random.default_rng(args.seed), 65536)

    print(f"NSM: {nsm} +/- {np.sqrt(var)}")  # noqa: T201


if __name__ == "__main__":
    main()
