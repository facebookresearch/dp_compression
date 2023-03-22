import numpy as np
from mechanisms import *
import pickle
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize and save MVU mechanisms.")
    parser.add_argument(
        "--method",
        default="tr",
        type=str,
        choices=["tr", "penalized"],
        help="which optimization method to use",
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
        default=3,
        type=int,
        help="number of input bits for the MVU mechanism",
    )
    parser.add_argument(
        "--lam",
        default=100,
        type=float,
        help="soft penalty for penalized solver"
    )
    parser.add_argument(
        "--num_iters",
        default=5,
        type=int,
        help="number of iterations for penalized solver"
    )
    parser.add_argument(
        "--dp_constraint",
        default="strict",
        type=str,
        choices=["strict", "metric-l1", "metric-l2"],
        help="type of DP constraint"
    )
    args = parser.parse_args()
    
    if args.method == "penalized":
        savedir = f"./sweep_eps_budget_penalized_lam{args.lam:0.1e}/"
    else:
        savedir = f"./sweep_eps_budget_tr/"
    os.makedirs(savedir, exist_ok=True)
    result_filename = os.path.join(
        savedir, f"mechanism_bin{args.input_bits}_bout{args.budget}_{args.dp_constraint}_eps{args.epsilon:0.2f}.pkl")
    
    if os.path.exists(result_filename):
        print(f"{result_filename} already exists, skipping")
        exit()
    
    if args.method == "penalized":
        mechanism = MVUMechanism(args.budget, args.epsilon, args.input_bits, method="penalized-lp", verbose=True,
                                 dp_constraint=args.dp_constraint, lam=args.lam, num_iters=args.num_iters)
    else:
        mechanism = MVUMechanism(args.budget, args.epsilon, args.input_bits, method="trust-region", verbose=True,
                                 dp_constraint=args.dp_constraint, init_method="uniform")
    with open(result_filename, "wb") as f:
        pickle.dump(mechanism, f)

