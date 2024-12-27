import argparse
import time
from pathlib import Path

import numpy as np

from lattice_quantizer.optimizer import SGDLatticeQuantizerOptimizer


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stochastic Gradient Descent method for lattice quantizers"
    )

    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        required=True,
        help="Dimension of the lattice quantizer",
    )

    parser.add_argument(
        "-mu0",
        "--initial_step_size",
        type=float,
        required=False,
        default=0.001,  # Medium:0.001; Fast: 0.005; Slow: 0.0005
        help="Initial step size for the stochastic gradient descent",
    )

    parser.add_argument(
        "-v",
        "--radio",
        type=int,
        required=False,
        default=500,  # Medium: 500; Fast: 200; Slow: 1000
        help="Ratio between initial and final step size",
    )

    parser.add_argument(
        "-T",
        "--steps",
        type=int,
        required=False,
        default=10000000,  # Medium: 10000000; Fast: 1000000; Slow: 100000000
        help="Number of steps for the stochastic gradient descent",
    )

    parser.add_argument(
        "-Tr",
        "--reduction_interval",
        type=int,
        required=False,
        default=100,
        help="Interval between consecutive reductions",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the results",
    )

    return parser.parse_args()


def main():
    args = parse_arg()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimzer = SGDLatticeQuantizerOptimizer(
        args.dimension,
        args.initial_step_size,
        args.radio,
        args.steps,
        args.reduction_interval,
    )

    B = optimzer.optimize()
    np.savetxt(
        output_dir / f"quantizer_{args.dimension}_{round(time.time())}.txt",
        B,
    )


if __name__ == "__main__":
    main()
