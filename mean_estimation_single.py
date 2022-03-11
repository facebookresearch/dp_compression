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
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


def compute_error(xs, mechanism, num_trials):
    squared_errors = np.zeros(xs.shape)
    for i in tqdm(range(len(xs))):
        x = xs[i]
        inputs = x * np.ones(num_trials)
        result = mechanism.decode(mechanism.privatize(inputs))
        squared_errors[i] = np.power(result - x, 2).mean()
    return squared_errors


def compute_error_1d(xs, mechanism, num_trials):
    means = np.zeros(xs.shape)
    squared_errors = np.zeros(xs.shape)
    for i in tqdm(range(len(xs))):
        x = np.array([xs[i]])
        for _ in range(num_trials):
            result = mechanism.decode(mechanism.privatize(x))
            means[i] += result / num_trials
            squared_errors[i] += np.power(result - x, 2) / num_trials
    return means, squared_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scalar distributed mean estimation experiment.")
    parser.add_argument(
        "--save_folder",
        default="figures",
        type=str,
        help="folder in which to save the figures",
    )
    parser.add_argument(
        "--mechanism_folder",
        default="sweep_eps_budget_tr",
        type=str,
        help="folder containing saved MVU mechanisms",
    )
    parser.add_argument(
        "--num_samples",
        default=int(1e5),
        type=int,
        help="number of samples",
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
    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)

    xs = np.linspace(-1, 1, 50)
    xs_normalized = (xs + 1) / 2
    
    print("Running RR")
    mechanism = RandomizedResponseMechanism(args.budget, args.epsilon)
    squared_errors_rr = 4 * compute_error(xs_normalized, mechanism, args.num_samples)
    
    print("Running RAPPOR")
    mechanism = RAPPORMechanism(args.budget, args.epsilon, args.budget)
    squared_errors_rappor = 4 * compute_error(xs_normalized, mechanism, args.num_samples)
    
    print("Running CLDP")
    mechanism = CLDPMechanism(args.epsilon, 1, 1, "linf")
    squared_errors_cldp = compute_error_1d(xs, mechanism, args.num_samples)[1]
    
    print("Running MVU b=%d" % args.budget)
    savefile = os.path.join(
        args.mechanism_folder, f"mechanism_bin{args.input_bits}_bout{args.budget}_strict_eps{args.epsilon:.2f}.pkl")
    if os.path.exists(savefile):
        with open(savefile, "rb") as file:
                mechanism = pickle.load(file)
    else:
        mechanism = MVUMechanism(args.budget, args.epsilon, args.input_bits, method="trust-region", init_method="uniform")
    squared_errors_mvu = 4 * compute_error(xs_normalized, mechanism, args.num_samples)
    
    print("Running MVU b=1")
    savefile = os.path.join(
        args.mechanism_folder, f"mechanism_bin{args.input_bits}_bout1_strict_eps{args.epsilon:.2f}.pkl")
    if os.path.exists(savefile):
        with open(savefile, "rb") as file:
                mechanism = pickle.load(file)
    else:
        mechanism = MVUMechanism(1, args.epsilon, args.input_bits, method="trust-region", init_method="uniform")
    squared_errors_mvu_1bit = 4 * compute_error(xs_normalized, mechanism, args.num_samples)
    
    plt.figure(figsize=(8,5))
    colors = sns.color_palette("deep")
    plt.plot(xs, squared_errors_rr, label='RR', color=colors[6], linewidth=3)
    plt.plot(xs, squared_errors_rappor, label='RAPPOR', color=colors[3], linewidth=3)
    plt.plot(xs, squared_errors_cldp, label='CLDP', color=colors[0], linewidth=3)
    plt.plot(xs, squared_errors_mvu_1bit, label='MVU ($b=1$)', color='lightgreen', linewidth=3)
    plt.plot(xs, squared_errors_mvu, label='MVU ($b=3$)', color=colors[2], linewidth=3)
    plt.plot(xs, np.ones(xs.shape) * 2 / args.epsilon**2, label='Laplace', color='k', linestyle='--', linewidth=3)

    plt.xlabel('$x$', fontsize=20)
    plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=20)
    plt.ylabel('Variance', fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid('on')
    plt.legend(loc='upper right', fontsize=20)
    plt.savefig("%s/dme_single_eps_%.2f.pdf" % (args.save_folder, args.epsilon), bbox_inches="tight")
