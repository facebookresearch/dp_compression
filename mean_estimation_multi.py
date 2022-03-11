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
from opacus.accountants.analysis.rdp import get_privacy_spent
import argparse
from utils import optimal_scaling_mvu, optimal_scaling_skellam
import torch.nn as nn

import sys
sys.path.append("private_prediction/")
from util import binary_search
from private_prediction import sensitivity_scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vector distributed mean estimation experiment.")
    parser.add_argument(
        "--save_folder",
        default="dme_results",
        type=str,
        help="folder in which to store results",
    )
    parser.add_argument(
        "--mechanism_folder",
        default="sweep_eps_budget_penalized_lam1.0e+02",
        type=str,
        help="folder containing saved MVU mechanisms",
    )
    parser.add_argument(
        "--trials",
        default=10,
        type=int,
        help="number of trials",
    )
    parser.add_argument(
        "--num_samples",
        default=10000,
        type=int,
        help="number of samples",
    )
    parser.add_argument(
        "--d",
        default=128,
        type=int,
        help="data dimensionality; must be a power of 2",
    )
    parser.add_argument(
        "--norm_type",
        default="l1",
        choices=["l1", "l2"],
        type=str,
        help="generate synthetic data under L1 or L2 norm bound",
    )
    parser.add_argument(
        "--epsilon",
        default=1,
        type=float,
        help="LDP epsilon",
    )
    parser.add_argument(
        "--skellam_budget",
        default=16,
        type=int,
        help="budget for the Skellam mechanism",
    )
    parser.add_argument(
        "--skellam_s",
        default=15,
        type=float,
        help="scaling factor for the Skellam mechanism",
    )
    parser.add_argument(
        "--mvu_budget",
        default=16,
        type=int,
        help="budget for the MVU mechanism",
    )
    parser.add_argument(
        "--mvu_input_bits",
        default=5,
        type=int,
        help="number of input bits for the MVU mechanism",
    )
    parser.add_argument(
        "--dither_tol",
        default=0.1,
        type=float,
        help="failure probability for conditional dithering",
    )
    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    
    if args.norm_type == "l1":
        p = 1
        # generate from uniform([0, 1]) then normalize
        xs = np.random.random((args.num_samples, args.d))
    else:
        p = 2
        # generate from uniform over positive quadrant of unit sphere
        xs = np.absolute(np.random.normal(0, 1, (args.num_samples, args.d)))
    xs /= np.maximum(np.linalg.norm(xs, p, 1), 1)[:, None]

    # CLDP
    mechanism_cldp = CLDPMechanism(args.epsilon, args.d, 1, args.norm_type)

    # Skellam
    mechanism = SkellamMechanism(args.skellam_budget, args.d, 1, 1, args.skellam_s)
    skellam_scale = optimal_scaling_skellam(xs, mechanism, args.skellam_s, args.dither_tol, p)
    
    mus = np.power(10, np.linspace(-2, 2, 100))
    orders = np.array(list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64)))
    for mu in mus:
        mechanism_skellam = SkellamMechanism(args.skellam_budget, args.d, 1, mu, args.skellam_s, p=p)
        rdp_const = mechanism_skellam.renyi_div(orders)
        epsilon_opt, _ = get_privacy_spent(orders=orders, rdp=rdp_const, delta=(1/(args.num_samples+1)))
        if epsilon_opt < args.epsilon:
            print("Optimal mu = %.2f" % mu)
            break
            
    # MVU
    epsilon = 2 * args.epsilon if args.norm_type == "l1" else 4 * args.epsilon
    mechanism = MultinomialSamplingMechanism(args.mvu_budget, epsilon, args.mvu_input_bits, norm_bound=0.5, p=None)
    mvu_scale = optimal_scaling_mvu(xs, mechanism, args.dither_tol, p)
    savefile = os.path.join(
        args.mechanism_folder, f"mechanism_bin{args.mvu_input_bits}_bout{args.mvu_budget}_metric-{args.norm_type}_eps{epsilon:.2f}.pkl")
    with open(savefile, "rb") as file:
        mechanism_mvu = pickle.load(file)
    mechanism_mvu.P /= mechanism_mvu.P.sum(1)[:, None]
    mechanism_mvu.norm_bound = 0.5
    mechanism_mvu.p = p
    
    # MVU mechanism for approximate DP
    epsilon = 2 * args.epsilon if args.norm_type == "l1" else 0.5 * args.epsilon
    savefile = os.path.join(
        args.mechanism_folder, f"mechanism_bin{args.mvu_input_bits}_bout{args.mvu_budget}_metric-l1_eps{epsilon:.2f}.pkl")
    with open(savefile, "rb") as file:
        mechanism_mvu_approx = pickle.load(file)
    mechanism_mvu_approx.P /= mechanism_mvu_approx.P.sum(1)[:, None]
    mechanism_mvu_approx.norm_bound = 0.5
    mechanism_mvu_approx.p = p

    squared_error_cldp = np.zeros(args.trials)
    squared_error_skellam = np.zeros(args.trials)
    squared_error_mvu = np.zeros(args.trials)
    squared_error_mvu_approx = np.zeros(args.trials)
    squared_error_baseline = np.zeros(args.trials)
    
    for k in tqdm(range(args.trials)):
        if args.norm_type == "l1":
            # generate from uniform([0, 1]) then normalize
            xs = np.random.random((args.num_samples, args.d))
        else:
            # generate from uniform over positive quadrant of unit sphere
            xs = np.absolute(np.random.normal(0, 1, (args.num_samples, args.d)))
        xs /= np.maximum(np.linalg.norm(xs, p, 1), 1)[:, None]
        mean = xs.mean(0)

        mean_cldp = np.zeros(xs.shape[1])
        for i in range(args.num_samples):
            x = xs[i]
            result = mechanism_cldp.decode(mechanism_cldp.privatize(x))
            mean_cldp += result / args.num_samples
        squared_error_cldp[k] = np.power(mean - mean_cldp, 2).mean()

        mean_skellam = np.zeros(xs.shape[1])
        for i in range(args.num_samples):
            x = skellam_scale * xs[i]
            output = mechanism_skellam.privatize(x)
            mean_skellam += output
        mean_skellam = np.mod(mean_skellam - mechanism_skellam.clip_min, mechanism_skellam.clip_max - mechanism_skellam.clip_min) + mechanism_skellam.clip_min
        mean_skellam = mechanism_skellam.decode(mean_skellam) / (skellam_scale * args.num_samples)
        squared_error_skellam[k] = np.power(mean - mean_skellam, 2).mean()

        mean_mvu = np.zeros(xs.shape[1])
        prepro = lambda z: mvu_scale * z
        prepro_inv = lambda z: z / mvu_scale
        for i in range(args.num_samples):
            x = prepro(xs[i])
            x = np.clip((x + 1) / 2, 0, 1)
            result = 2 * mechanism_mvu.decode(mechanism_mvu.privatize(x)) - 1
            mean_mvu += prepro_inv(result) / args.num_samples
        squared_error_mvu[k] = np.power(mean - mean_mvu, 2).mean()

        mean_mvu_approx = np.zeros(xs.shape[1])
        for i in range(args.num_samples):
            x = prepro(xs[i])
            x = np.clip((x + 1) / 2, 0, 1)
            result = 2 * mechanism_mvu_approx.decode(mechanism_mvu_approx.privatize(x)) - 1
            mean_mvu_approx += prepro_inv(result) / args.num_samples
        squared_error_mvu_approx[k] = np.power(mean - mean_mvu_approx, 2).mean()

        if args.norm_type == "l1":
            mean_baseline = (xs + np.random.laplace(0, 1 / args.epsilon, size=xs.shape)).mean(0)
        else:
            std = 1 / sensitivity_scale(args.epsilon, 1/(args.num_samples+1), None, None, None,
                                        "advanced_gaussian", chaudhuri=False)
            mean_baseline = (xs + np.random.normal(0, std, size=xs.shape)).mean(0)
        squared_error_baseline[k] = np.power(mean - mean_baseline, 2).mean()
        
    savefile = "%s/dme_multi_%s_d_%d_samples_%d_eps_%.2f_skellam_%d_%.2f_mvu_%d_%d.npz" % (
        args.save_folder, args.norm_type, args.d, args.num_samples, args.epsilon, args.skellam_budget, args.skellam_s, args.mvu_budget, args.mvu_input_bits)
    np.savez(savefile, squared_error_cldp=squared_error_cldp, squared_error_skellam=squared_error_skellam,
             squared_error_mvu=squared_error_mvu, squared_error_mvu_approx=squared_error_mvu_approx,
             squared_error_baseline=squared_error_baseline)
