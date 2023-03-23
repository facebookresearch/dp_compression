#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
import sys
import math
import scipy.optimize as optimize
from utils import optimal_scaling_integer
from hadamard_transform import pad_to_power_of_2, randomized_hadamard_transform, inverse_randomized_hadamard_transform
        

class SkellamMechanismPyTorch:
    '''
    Skellam mechanism from https://arxiv.org/pdf/2110.04995.pdf.
    '''
    
    def __init__(self, budget, d, norm_bound, mu, device, num_clients=1, s=None):
        self.budget = budget
        self.d = d
        self.expanded_d = int(math.pow(2, math.ceil(math.log2(d))))
        self.norm_bound = norm_bound
        self.mu = mu
        if s is None:
            self.s = self.compute_s(num_clients)
            # print("s = %.2f" % self.s)
        else:
            self.s = s
        self.scale = optimal_scaling_integer(self.expanded_d, self.s * norm_bound, math.exp(-0.5), tol=1e-3)
        if self.scale == 0:
            raise RuntimeError("Did not find suitable scale factor; try increasing communication budget")
        self.clip_min = -int(math.pow(2, budget - 1))
        self.clip_max = int(math.pow(2, budget - 1)) - 1
        self.device = device
        self.seed = None
        return
    
    def compute_s(self, num_clients, k=3, rho=1, DIV_EPSILON=1e-22):
        """
        Adapted from https://github.com/google-research/federated/blob/master/distributed_dp/accounting_utils.py
        """
        def mod_min(gamma):
            var = rho / self.d * (num_clients * self.norm_bound)**2
            var += (gamma**2 / 4 + self.mu) * num_clients
            return k * math.sqrt(var)

        def gamma_opt_fn(gamma):
            return (math.pow(2, self.budget) - 2 * mod_min(gamma) / (gamma + DIV_EPSILON))**2

        gamma_result = optimize.minimize_scalar(gamma_opt_fn)
        if not gamma_result.success:
            raise ValueError('Cannot compute scaling factor.')
        return 1. / gamma_result.x
    
    def renyi_div(self, alphas, l1_norm_bound=None, l2_norm_bound=None):
        """
        Computes Renyi divergence of the Skellam mechanism.
        """
        if l2_norm_bound is None:
            l2_norm_bound = self.norm_bound
        if l1_norm_bound is None:
            l1_norm_bound = self.norm_bound * min(math.sqrt(self.expanded_d), self.norm_bound)
        epsilons = np.zeros(alphas.shape)
        B1 = 3 * l1_norm_bound / (2 * self.s ** 3 * self.mu ** 2)
        B2 = 3 * l1_norm_bound / (2 * self.s * self.mu)
        for i in range(len(alphas)):
            alpha = alphas[i]
            epsilon = alpha * self.norm_bound ** 2 / (2 * self.mu)
            B3 = (2 * alpha - 1) * self.norm_bound ** 2 / (4 * self.s ** 2 * self.mu ** 2)
            epsilons[i] = epsilon + min(B1 + B3, B2)
        return epsilons
    
    def dither(self, x):
        k = torch.floor(x).to(self.device)
        prob = 1 - (x - k)
        while True:
            output = k + (torch.rand(k.shape).to(self.device) > prob)
            if output.norm() <= self.s * self.norm_bound:
                break
        return output.long()
    
    def privatize(self, x, same_rotation_batch=False):
        assert torch.all(x.norm(2, 1) <= self.norm_bound * (1 + 1e-4))  # add some margin due to clipping rounding issues
        assert x.size(1) == self.d
        prng = torch.Generator(device=self.device)
        self.seed = prng.seed()
        x = randomized_hadamard_transform(pad_to_power_of_2(x), prng.manual_seed(self.seed), same_rotation_batch)
        z = torch.zeros(x.size()).long().to(self.device)
        for i in range(x.shape[0]):
            z[i] = self.dither(self.s * x[i])
        dist = torch.distributions.poisson.Poisson(self.s**2 * self.mu)
        z += (dist.sample(z.size()) - dist.sample(z.size())).long().to(self.device)
        z = torch.remainder(z - self.clip_min, self.clip_max - self.clip_min) + self.clip_min
        return z
    
    def decode(self, z, same_rotation_batch=False):
        assert self.seed is not None, "Must call privatize before decode."
        prng = torch.Generator(device=self.device)
        x = inverse_randomized_hadamard_transform(z.float(), prng.manual_seed(self.seed), same_rotation_batch) / self.s
        self.seed = None
        return x[:, :self.d]


class MVUMechanismPyTorch:
    
    def __init__(self, input_bits, budget, epsilon, P, alpha, norm_bound, device):
        self.budget = budget
        self.scale = 1
        self.epsilon = epsilon
        self.input_bits = input_bits
        self.P = P.to(device)
        self.alpha = alpha.to(device)
        self.norm_bound = norm_bound
        self.device = device
        return
    
    def dither(self, x):
        assert torch.all(x >= 0) and torch.all(x <= 1)
        B = 2 ** self.input_bits
        k = torch.floor((B-1) * x).to(self.device)
        prob = 1 - (B-1) * (x - k/(B-1))
        while True:
            output = k + (torch.rand(k.shape).to(self.device) > prob)
            if (output / (B-1) - 0.5).norm() <= self.norm_bound:
                break
        return output.long()
    
    def privatize(self, x):
        z = x.new_zeros(x.size()).long()
        for i in range(x.size(0)):
            z[i] = self.dither(x[i])
        z = z.flatten()
        B = 2 ** self.input_bits
        range_B = torch.arange(B).long().to(self.device)
        output = torch.zeros(z.shape).long().to(self.device)
        for i in range(len(range_B)):
            mask = z.eq(range_B[i])
            if mask.sum() > 0:
                output[mask] = torch.multinomial(self.P[i], mask.sum(), replacement=True)
        return output
    
    def decode(self, k):
        assert k.min() >= 0 and k.max() < 2 ** self.budget
        return self.alpha[k]
    
    
class IMVUMechanismPyTorch:
    
    def __init__(self, input_bits, budget, P, alpha, device):
        self.input_bits = input_bits
        self.budget = budget
        self.scale = 1
        self.P = P.to(device)
        self.eta = self.P.log()
        self.alpha = alpha.to(device)
        self.device = device
        return
    
    def get_etas(self, x):
        assert torch.all(x >= 0) and torch.all(x <= 1)
        B = 2**self.input_bits
        k = torch.floor((B-1) * x).long()
        coeff1 = k + 1 - x * (B-1)
        coeff2 = x * (B-1) - k
        eta1 = self.eta[k]
        eta2 = self.eta[k+(coeff2>0)]
        return coeff1[:, None] * eta1 + coeff2[:, None] * eta2
    
    def privatize(self, x, batch_size=int(1e8)):
        x = x.flatten()
        output = []
        num_batch = int(math.ceil(len(x) / float(batch_size)))
        for i in range(num_batch):
            lower = i * batch_size
            upper = min((i+1) * batch_size, len(x))
            z = x[lower:upper]
            P = F.softmax(self.get_etas(z), dim=1)
            if P.size(1) == 2:
                output.append(torch.bernoulli(P[:, 1]))
            else:
                output.append(torch.multinomial(P, 1, replacement=True).squeeze())
        output = torch.cat(output, 0).long().to(self.device)
        return output
    
    def decode(self, k):
        assert k.min() >= 0 and k.max() < 2 ** self.budget
        return self.alpha[k]
    
