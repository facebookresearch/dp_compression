#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mechanisms import *
import numpy as np
from tqdm import tqdm
from utils import renyi_div_bound_lp
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Renyi divergence for the MVU mechanism.")
    parser.add_argument(
        "--mechanism_folder",
        default="sweep_eps_budget_penalized_lam1.0e+02",
        type=str,
        help="folder containing saved optimal mechanisms",
    )
    parser.add_argument(
        "--d",
        default=128,
        type=int,
        help="data dimensionality; must be a power of 2",
    )
    parser.add_argument(
        "--Delta",
        default=0.5,
        type=float,
        help="input sensitivity Delta"
    )
    parser.add_argument(
        "--epsilon",
        default=1,
        type=float,
        help="LDP epsilon",
    )
    parser.add_argument(
        "--budget",
        default=3,
        type=int,
        help="budget for the MVU mechanism",
    )
    parser.add_argument(
        "--input_bits",
        default=5,
        type=int,
        help="number of input bits for the MVU mechanism",
    )
    parser.add_argument(
        "--dp_constraint",
        default="strict",
        type=str,
        choices=["strict", "metric-l1", "metric-l2"],
        help="type of DP constraint"
    )
args = parser.parse_args()

alphas = np.array(list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64)))
savefile = os.path.join(args.mechanism_folder,
                        f"mechanism_bin{args.input_bits}_bout{args.budget}_{args.dp_constraint}_eps{args.epsilon:.2f}.pkl")
with open(savefile, "rb") as file:
    mechanism = pickle.load(file)
renyi_div_bound = renyi_div_bound_lp(alphas, args.d, mechanism.P, args.Delta)
output_file = f"renyi_div_bin{args.input_bits}_bout{args.budget}_{args.dp_constraint}_eps{args.epsilon:.2f}_Delta{args.Delta:.2f}.npz"
np.savez(os.path.join(args.mechanism_folder, output_file), alphas=alphas, renyi_div_bound=renyi_div_bound)
