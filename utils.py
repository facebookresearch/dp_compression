#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from tqdm import tqdm
from scipy.optimize import linprog

import sys
sys.path.append("private_prediction/")
from util import binary_search

    
def _log_add(logx, logy):
    a, b = np.minimum(logx, logy), np.maximum(logx, logy)
    eq_zero = (a == -np.inf)
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    result = np.log(np.exp(logx - logy) + 1) + logy
    result[eq_zero] = b[eq_zero]
    return result


def log_sum(log_zs):
    shape = log_zs.sum(-1).shape
    log_zs = log_zs.reshape(-1, log_zs.shape[-1])
    result = log_zs[:, 0]
    for i in range(1, log_zs.shape[-1]):
        result = _log_add(result, log_zs[:, i])
    return result
    
    
def renyi_div_stable(P, order):
    """
    Computes (log) Renyi divergence using stable log-space computation
    """
    assert order > 1, "Renyi divergence order must be > 1"
    assert np.all(P >= 0)
    log_P = np.log(P)
    log_Q = np.copy(log_P)
    log_Q[log_Q == -np.inf] = 0
    log_zs = order * log_P[:, None, :] - (order - 1) * log_Q[None, :, :]
    return log_sum(log_zs) / (order - 1)


def renyi_div_bound_lp(alphas, d, P, Delta, greedy=False, verbose=False):
    """
    Computes LP relaxation upper bound for Renyi divergence of MVU mechanism.
    If greedy=True, always compute greedy solution.
    """
    B = P.shape[0]
    bounds = np.zeros(alphas.shape)
    for i in tqdm(range(alphas.shape[0])):
        alpha = alphas[i]
        D = renyi_div_stable(P, alpha)
        D[D < 0] = 0
        irange = np.arange(0, B) / (B-1)
        # add identity to cost to avoid division by zero
        cost = (np.power(irange[:, None] - irange[None, :], 2) + np.eye(B)).flatten()
        opt = (D / cost).argmax()
        max_div, cost_per_dim = D[opt], cost[opt]
        if d * cost_per_dim < Delta**2 and (not greedy):
            c = -np.tile(D, d)
            A = np.tile(np.power(irange[:, None] - irange[None, :], 2).flatten(), d)[None, :]
            A = np.concatenate([A, np.kron(np.eye(d), np.ones((1, B*B)))], 0)
            b = np.array([0.25] + [1] * d)
            res = linprog(c, A_ub=A, b_ub=b, options={"disp": verbose})
            bounds[i] = -res.fun
        else:
            # compute bound using optimal greedy solution
            num_items = Delta**2 / cost_per_dim
            bounds[i] = max_div * num_items
    return bounds


def optimal_scaling_mvu(samples, mechanism, conf, p=2):
    """
    Computes optimal scaling factor for MVU mechanism to ensure that vector norm after
    dithering does not increase.
    """
    B = 2 ** mechanism.input_bits
    def constraint(t):
        x = t * samples
        x = np.clip((x+1)/2, 0, 1)
        z = mechanism.dither(x, mechanism.input_bits) / (B-1)
        z = 2*z - 1
        norms = np.linalg.norm(z, p, 1)
        return (norms > 1).astype(float).mean() < conf
    return binary_search(lambda t: t, constraint, 0, 1, tol=1e-3)


def optimal_scaling_skellam(samples, mechanism, s, conf, p=2):
    """
    Computes optimal scaling factor for Skellam mechanism to ensure that vector norm after
    dithering does not increase.
    """
    def constraint(t):
        x = t * samples
        z = mechanism.dither(s * x) / s
        norms = np.linalg.norm(z, p, 1)
        return (norms > 1).astype(float).mean() < conf
    return binary_search(lambda t: t, constraint, 0, 1, tol=1e-3)
    