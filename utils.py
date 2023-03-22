import numpy as np
from tqdm import tqdm
from scipy.optimize import linprog
from scipy.special import softmax
import math
import torch

import sys
sys.path.append("private_prediction/")
from util import binary_search
sys.path.append("fastwht/python/")
from hadamard import *
    
    
class FWHTRandomProjector:
    '''
    Fast random projector using the Fast Walsh-Hadamard Transform.
    
    Arguments:
        num_parameters:    Number of input data dimensions
        rank:              Rank of projection
        seed:              Random seed to control the projection.
                           Default None uses a different random seed each time.
    '''
    
    def __init__(self, num_parameters, expanded_dim):
        self.num_parameters = num_parameters
        self.expanded_dim = expanded_dim
        self.D = np.sign(np.random.normal(0, 1, self.expanded_dim))
        
    def project(self, x):
        x_expanded = np.zeros(self.expanded_dim)
        x_expanded[:len(x)] = x
        z = fastwht(x_expanded * self.D, order='hadamard') * math.sqrt(float(x_expanded.shape[0]))
        return z
    
    def inverse(self, x):
        out_expanded = fastwht(x, order='hadamard') * self.D * math.sqrt(float(x.shape[0]))
        return out_expanded[:self.num_parameters]

    
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
    return np.reshape(log_sum(log_zs) / (order - 1), (P.shape[0], P.shape[0]))


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


def post_rounding_l2_norm_bound(d, l2_norm_bound, beta):
    """
    Function for computing vector norm bound after quantizing to the integer grid.
    Adapted from https://github.com/google-research/federated/blob/master/distributed_dp/compression_utils.py
    """
    bound1 = l2_norm_bound + math.sqrt(d)
    squared_bound2 = l2_norm_bound**2 + 0.25 * d
    squared_bound2 += (math.sqrt(2.0 * math.log(1.0 / beta)) * (l2_norm_bound + 0.5 * math.sqrt(d)))
    bound2 = math.sqrt(squared_bound2)
    # bound2 is inf if beta = 0, in which case we fall back to bound1.
    return min(bound1, bound2)


def optimal_scaling_integer(d, l2_norm_bound, beta, tol=1e-3):
    """
    Computes optimal scaling factor for DDG/Skellam mechanism to ensure that vector norm after
    dithering does not increase.
    """
    def constraint(t):
        if t == 0:
            return True
        quantized_norm = post_rounding_l2_norm_bound(d, t, beta)
        return quantized_norm <= l2_norm_bound + 1e-6
    opt_norm = binary_search(lambda t: t, constraint, 0, l2_norm_bound, tol=tol)
    return opt_norm / l2_norm_bound


def max_divergence_bound(log_P, precision=1e-4):
    """
    Computes L1 metric DP additive factor for the interpolated MVU mechanism.
    """
    B = log_P.shape[0]
    epsilon_bound = np.zeros(B - 1)
    for i in range(log_P.shape[0] - 1):
        theta = log_P[i+1] - log_P[i]
        xs = np.arange(0, 1 + precision/2, precision)
        etas = xs[:, None] * log_P[None, i+1] + (1 - xs)[:, None] * log_P[None, i]
        func = abs(softmax(etas, axis=1) @ theta) * (B - 1)
        epsilon_bound[i] = func.max()
    return epsilon_bound.max()


def fisher_information(bs, eta1, eta2):
    """
    Computes Fisher information of the interpolated MVU mechanism for a range of coefficients.
    """
    etas = (1 - bs)[:, None] * eta1[None, :] + bs[:, None] * eta2[None, :]
    P = softmax(etas, axis=1)
    fi = P @ np.power(eta2 - eta1, 2) - np.power(P @ (eta2 - eta1), 2)
    return fi


def fisher_information_bound(p, precision=1e-4):
    """
    Asserts the condition that maximum FI occurs in range [0,1] and returns this maximum.
    If condition fails, return -1.
    """
    
    # sampling probability vectors must be symmetric about 0.5(!)
    eta1 = np.log(p)
    eta2 = eta1[::-1]
    
    # compute maximum FI within range [0,1]
    bs = np.arange(0, 1+precision/2, precision)
    fi_max = fisher_information(bs, eta1, eta2).max()
    
    # compute range [a_min, a_max]
    j_max = np.argmax(eta2 - eta1)
    eta_sq_max = np.power(eta2 - eta1, 2)[j_max]
    c = fi_max / (4 * eta_sq_max)
    p = (1 + math.sqrt(1 - 4*c)) / 2
    assert p >= 0.5
    a_max = 1
    while True:
        eta = (1-a_max) * eta1 + a_max * eta2
        p_eta = softmax(eta)
        if p_eta[j_max] >= p:
            break
        else:
            a_max += 1
    a_min = 1 - a_max
    
    # compute maximum FI within range [a_min, a_max]
    bs = np.arange(a_min, a_max + precision/2, precision)
    fi = fisher_information(bs, eta1, eta2)
    return fi.max()


def consolidate(mechanism, tol=1e-8):
    """
    Consolidates the sampling probability matrix P to remove redundant columns.
    Must be called prior to computing the Fisher information to avoid infinite loop!
    """
    digits = int(-math.log10(tol))
    uniques = np.unique(mechanism.alpha.round(digits))
    P_new = []
    alpha_new = []
    for i in range(len(uniques)):
        indices = np.arange(0, len(mechanism.alpha))[np.isclose(mechanism.alpha, uniques[i])]
        p = mechanism.P[:, indices].sum(1)
        P_new.append(p[:, None])
        alpha_new.append(mechanism.alpha[indices[0]])
    P_new = np.concatenate(P_new, 1)
    alpha_new = np.hstack(alpha_new)
    return P_new, alpha_new


def params_to_vec(model, return_type="param"):
    '''
    Helper function that concatenates model parameters or gradients into a single vector.
    '''
    vec = []
    for param in model.parameters():
        if return_type == "param":
            vec.append(param.data.view(1, -1))
        elif return_type == "grad":
            vec.append(param.grad.view(1, -1))
        elif return_type == "grad_sample":
            if hasattr(param, "grad_sample"):
                vec.append(param.grad_sample.view(param.grad_sample.size(0), -1))
            else:
                print("Error: Per-sample gradient not found")
                sys.exit(1)
    return torch.cat(vec, dim=1).squeeze()


def set_grad_to_vec(model, vec):
    '''
    Helper function that sets the model's gradient to a given vector.
    '''
    model.zero_grad()
    for param in model.parameters():
        size = param.data.view(1, -1).size(1)
        param.grad = vec[:size].view_as(param.data).clone()
        vec = vec[size:]
    return
    