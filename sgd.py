import argparse
import datetime
import logging
from pathlib import Path

import numpy as np


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
        "-m",
        "--mode",
        type=str,
        default="medium",
        help="Mode of the optimization",
    )

    parser.add_argument(
        "-mu0",
        "--initial_step_size",
        type=float,
        required=False,
        help="Initial step size for the stochastic gradient descent",
    )

    parser.add_argument(
        "-v",
        "--radio",
        type=int,
        required=False,
        help="Ratio between initial and final step size",
    )

    parser.add_argument(
        "-T",
        "--steps",
        type=int,
        required=False,
        help="Number of steps for the stochastic gradient descent",
    )

    parser.add_argument(
        "-Tr",
        "--reduction_interval",
        type=int,
        required=False,
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
    output_suffix = f"{args.dimension}"

    if args.mode == "medium":
        initial_step_size = 0.001
        radio = 500
        steps = 10_000_000
        reduction_interval = 100
        output_suffix += "_medium"
    elif args.mode == "fast":
        initial_step_size = 0.005
        radio = 200
        steps = 1_000_000
        reduction_interval = 100
        output_suffix += "_fast"
    elif args.mode == "slow":
        initial_step_size = 0.0005
        radio = 1000
        steps = 100_000_000
        reduction_interval = 100
        output_suffix += "_slow"
    else:
        logging.fatal(
            f"Mode {args.mode} not recognized, accepted values are 'medium', 'fast' and 'slow'"
        )
        return

    if args.initial_step_size is not None:
        initial_step_size = args.initial_step_size
        output_suffix += f"_mu0[{args.initial_step_size}]"

    if args.radio is not None:
        radio = args.radio
        output_suffix += f"_v[{args.radio}]"

    if args.steps is not None:
        steps = args.steps
        output_suffix += f"_T[{args.steps}]"

    if args.reduction_interval is not None:
        reduction_interval = args.reduction_interval
        output_suffix += f"_Tr[{args.reduction_interval}]"

    from lattice_quantizer.criteria.nsm import nsm_cpu
    from lattice_quantizer.optimizer import SGDLatticeQuantizerOptimizer

    optimzer = SGDLatticeQuantizerOptimizer(
        args.dimension,
        initial_step_size,
        radio,
        steps,
        reduction_interval,
    )

    result = optimzer.optimize()
    nsm, var = nsm_cpu(result, 100000, np.random.default_rng(), 1024)
    output_suffix += f"_nsm[{np.format_float_positional(nsm, 5)}]_var[{np.format_float_scientific(var, 3)}]"
    np.savetxt(
        output_dir
        / f"quantizer_{output_suffix}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt",
        result,
    )


if __name__ == "__main__":
    main()
