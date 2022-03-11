#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import numpy as np
from abc import abstractmethod
import scipy
from scipy.stats import skellam
import scipy.optimize as optimize
from scipy import sparse
from scipy.special import gamma
import cvxpy
import pdb
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm
import os
from datetime import datetime
import pickle


class CLDPMechanism:
    '''
    LDP mechanisms from https://arxiv.org/pdf/2008.07180.pdf.
    '''
    
    def __init__(self, epsilon, d, norm_bound, norm_type):
        self.epsilon = epsilon
        if norm_type == "l1":
            self.d = int(math.pow(2, math.ceil(math.log2(d))))
        else:
            self.d = d
        self.norm_bound = norm_bound
        self.norm_type = norm_type
        
    def privatize_l1(self, x):
        assert np.linalg.norm(x, 1) <= self.norm_bound + 1e-8
        assert len(x) <= self.d
        z = np.zeros(self.d)
        z[:len(x)] = x
        H = scipy.linalg.hadamard(self.d)
        z = H @ z / math.sqrt(self.d)
        C = (math.exp(self.epsilon) - 1) / (math.exp(self.epsilon) + 1)
        idx = random.randint(0, self.d - 1)
        U = np.sign(0.5 + C * math.sqrt(self.d) * z[idx] / (2 * self.norm_bound) - random.random())
        return (idx, U)
    
    def decode_l1(self, z):
        C_inv = (math.exp(self.epsilon) + 1) / (math.exp(self.epsilon) - 1)
        H = scipy.linalg.hadamard(self.d)
        return z[1] * self.norm_bound * C_inv * H[:, z[0]]
    
    def quantize_l2(self, x):
        norm = np.linalg.norm(x, 1)
        sign = np.sign((1 + norm) / (2 * self.norm_bound * math.sqrt(self.d)) - random.random())
        x = sign * x / norm
        if len(x) > 1:
            y = np.random.multinomial(1, np.absolute(x), (self.d,))
        else:
            y = np.ones((self.d, 1)).astype(int)
        return y, np.sign(x[np.argmax(y, 1)])
    
    def privatize_l2(self, x):
        assert np.linalg.norm(x, 2) <= self.norm_bound + 1e-8
        d = len(x)
        norm = np.linalg.norm(x, 2)
        sign = np.sign(0.5 + norm / (2 * self.norm_bound) - random.random())
        x = sign * self.norm_bound * x / norm
        U_sign = np.sign(math.exp(self.epsilon) / (math.exp(self.epsilon) + 1) - random.random())
        C_inv = (math.exp(self.epsilon) + 1) / (math.exp(self.epsilon) - 1)
        M = self.norm_bound * d * math.sqrt(math.pi) * C_inv * gamma((d-1)/2 + 1) / (2 * gamma(d/2 + 1))
        while True:
            z = np.random.normal(0, 1, (d,))
            z = M * z / np.linalg.norm(z)
            if sum(z * x) * U_sign > 0:
                break
        return self.quantize_l2(z)
    
    def decode_l2(self, z):
        return self.norm_bound * (z[0] * z[1][:, None]).sum(0)
    
    def privatize_linf(self, x):
        assert np.absolute(x).max() <= self.norm_bound + 1e-8
        C = (math.exp(self.epsilon) - 1) / (math.exp(self.epsilon) + 1)
        idx = random.randint(0, len(x) - 1)
        U = np.sign(0.5 + C * x[idx] / (2 * self.norm_bound) - random.random())
        return (idx, U)
    
    def decode_linf(self, z):
        C_inv = (math.exp(self.epsilon) + 1) / (math.exp(self.epsilon) - 1)
        e = np.zeros(self.d)
        e[z[0]] = 1
        return z[1] * self.norm_bound * self.d * C_inv * e
    
    def privatize(self, x):
        if self.norm_type == "l1":
            return self.privatize_l1(x)
        elif self.norm_type == "l2":
            return self.privatize_l2(x)
        elif self.norm_type == "linf":
            return self.privatize_linf(x)
        else:
            raise RuntimeError("Unsupported norm type: " + str(self.norm_type))
    
    def decode(self, x):
        if self.norm_type == "l1":
            return self.decode_l1(x)
        elif self.norm_type == "l2":
            return self.decode_l2(x)
        elif self.norm_type == "linf":
            return self.decode_linf(x)
        else:
            raise RuntimeError("Unsupported norm type: " + str(self.norm_type))
            
            
class SkellamMechanism:
    '''
    Skellam mechanism from https://arxiv.org/pdf/2110.04995.pdf.
    '''
    
    def __init__(self, budget, d, norm_bound, mu, s, p=None):
        self.budget = budget
        self.d = int(math.pow(2, math.ceil(math.log2(d))))
        self.norm_bound = norm_bound
        self.mu = mu
        self.s = s
        self.p = p
        self.clip_min = -int(math.pow(2, budget - 1))
        self.clip_max = int(math.pow(2, budget - 1)) - 1
        self.D = np.sign(np.random.normal(0, 1, (d,)))
        return
    
    def renyi_div(self, alphas, l1_norm_bound=None, l2_norm_bound=None):
        """
        Computes Renyi divergence of the Skellam mechanism.
        """
        if l2_norm_bound is None:
            l2_norm_bound = self.norm_bound
        if l1_norm_bound is None:
            l1_norm_bound = self.norm_bound * min(math.sqrt(self.d), self.norm_bound)
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
        k = np.floor(x)
        prob = 1 - (x - k)
        while True:
            output = k + (np.random.random(k.shape) > prob)
            if self.p is None or np.linalg.norm(output, self.p) <= self.s * self.norm_bound:
                break
        return output.astype(int)
    
    def privatize(self, x):
        assert np.linalg.norm(x, 2) <= self.norm_bound + 1e-8
        assert len(x) <= self.d
        z = np.zeros(self.d)
        z[:len(x)] = self.s * x
        H = scipy.linalg.hadamard(self.d) / math.sqrt(self.d)
        z = H @ (self.D * z)
        z = self.dither(z)
        z += skellam.rvs(self.s ** 2 * self.mu, self.s ** 2 * self.mu, size=z.shape)
        z = np.mod(z - self.clip_min, self.clip_max - self.clip_min) + self.clip_min
        return z
    
    def decode(self, z):
        H = scipy.linalg.hadamard(self.d) / math.sqrt(self.d)
        return (self.D * (H.T @ z)) / self.s


class CompressedMechanism:
    
    def __init__(self, budget, epsilon):
        self.budget = budget
        self.epsilon = epsilon
        return
    
    def dither(self, x, b, p=None):
        """
        Given x in [0,1], return a randomized dithered output in {0, 1, ..., 2^b - 1}.
        """
        assert np.all(x >= 0) and np.all(x <= 1)
        B = 2 ** b
        k = np.floor((B-1) * x)
        prob = 1 - (B-1) * (x - k/(B-1))
        k += np.random.random(k.shape) > prob
        return k.astype(int)
    
    @abstractmethod
    def privatize(self, x):
        """
        Privatizes a vector of values in [0,1] to binary vectors.
        """
        return
    
    @abstractmethod
    def decode(self, l):
        """
        Decodes binary vectors to an array of real values.
        """
        return
        
        
class RandomizedResponseMechanism(CompressedMechanism):
    
    def _privatize_bit(self, x, epsilon):
        """
        Privatizes a vector of bits using the binary randomized response mechanism.
        """
        assert np.all(np.logical_or(x == 0, x == 1))
        prob = 1 / (1 + math.exp(-epsilon))
        mask = np.random.random(x.shape) > prob
        z = np.logical_xor(mask, x).astype(int)
        return z
    
    def binary_repr(self, x):
        """
        Converts an array of integers to a 2D array of bits using binary representation.
        """
        l = [np.fromiter(map(int, np.binary_repr(a, width=self.budget)), int) for a in x]
        return np.stack(l, 0)
    
    def int_repr(self, l):
        """
        Converts a 2D array of bits into an array of integers using binary representation.
        """
        powers = np.power(2, np.arange(self.budget-1, -0.5, -1))
        return l.dot(powers)
    
    def privatize(self, x):
        z = self.dither(x, self.budget)
        l = self.binary_repr(z)
        l = self._privatize_bit(l, float(self.epsilon/self.budget))
        return l
    
    def decode(self, l):
        assert l.shape[1] == self.budget
        a_0 = -1 / (math.exp(self.epsilon/self.budget) - 1)
        a_1 = math.exp(self.epsilon/self.budget) / (math.exp(self.epsilon/self.budget) - 1)
        l = a_0 + l * (a_1 - a_0)
        return self.int_repr(l) / (2**self.budget - 1)
    
    
class MultinomialSamplingMechanism(CompressedMechanism):
    
    def __init__(self, budget, epsilon, input_bits, norm_bound, p, **kwargs):
        """
        Parent class that supports sampling from a 2^budget-dimensional distribution defined by
        a sampling probability matrix P and an output vector alpha.
        
        Arguments:
        budget     - Number of bits in the output.
        epsilon    - DP/metric-DP parameter epsilon.
        input_bits - Number of bits in the quantized input.
        norm_bound - A priori bound on the norm of the input before quantization; ignored if p=None.
        p          - Which p-norm to use for the norm bound parameter.
        """
        super().__init__(budget, epsilon)
        self.input_bits = input_bits
        self.norm_bound = norm_bound
        self.p = p
        result = self.optimize(**kwargs)
        if result is not None:
            self.P, self.alpha = result[0], result[1]
        return
    
    def dither(self, x, b):
        """
        Dithers x coordinate-wise to a grid of size 2^b.
        If self.p is set, perform rejection sampling until dithered vector does not exceed self.norm_bound.
        """
        assert np.all(x >= 0) and np.all(x <= 1)
        B = 2 ** b
        k = np.floor((B-1) * x)
        prob = 1 - (B-1) * (x - k/(B-1))
        while True:
            output = k + (np.random.random(k.shape) > prob)
            if self.p is None or np.linalg.norm(output / (B-1) - 0.5, self.p) <= self.norm_bound:
                break
        return output.astype(int)
    
    @abstractmethod
    def optimize(self, **kwargs):
        """
        Optimizes self.P and self.alpha for multinomial sampling.
        """
        return
    
    def privatize(self, x):
        z = self.dither(x, self.input_bits)
        B = 2**self.budget
        range_B = np.arange(B).astype(int)
        z = np.array([np.random.choice(range_B, p=self.P[a]) for a in z])
        return z
    
    def decode(self, z):
        assert z.max() < 2**self.budget
        return self.alpha[z.astype(int)]
    
    def mse_and_bias_squared(self, P=None, alpha=None):
        """
        Evaluate MSE loss and bias squared.
        """
        if P is None and alpha is None:
            P = self.P
            alpha = self.alpha
        B = 2 ** self.input_bits
        target = np.arange(0, 1+1/B, 1/(B-1))
        mse_loss = (P * np.power(target[:, None] - alpha[None, :], 2)).sum(1).mean()
        bias_sq = np.power(P @ alpha - target, 2).mean()
        return mse_loss, bias_sq
    

class RAPPORMechanism(MultinomialSamplingMechanism):
    
    def __init__(self, budget, epsilon, input_bits, norm_bound=0.5, p=None, **kwargs):
        super().__init__(budget, epsilon, budget, norm_bound, p, **kwargs)     # ignores input bits
        return
    
    def optimize(self):
        B = 2**self.budget
        prob = B / (B + math.exp(self.epsilon) - 1)
        P = prob / B * np.ones((B, B)) + (1 - prob) * np.eye(B)
        target = np.arange(0, 1+1/B, 1/(B-1))
        alpha = np.linalg.solve(P, target)
        return P, alpha
    
    
class MVUMechanism(MultinomialSamplingMechanism):
    
    def __init__(self, budget, epsilon, input_bits, norm_bound=0.5, p=None, **kwargs):
        super().__init__(budget, epsilon, input_bits, norm_bound, p, **kwargs)
        return
    
    # ==================================
    # Functions used by multiple methods
    # ==================================
    def _get_dp_constraint_matrix(self, dp_constraint):
        """
        Returns a sparse matrix D with shape (B*B*(B-1), B*B) such that D @ p
        corresponds to the left-hand side of the DP constraints on P,
        where p = P.reshape(B*B) and B = 2 ** self.budget.

        The final constriants are D @ p <= 0.

        Each row of D bounds the probability ratio between P_{i,j} and P_{i',j}
        by e^epsilon for i != i'. For metric DP, the ratio is e^{epsilon * abs(i - i')}.
        """
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        if dp_constraint == "strict":
            data = -math.exp(self.epsilon) * np.ones(B_in*(B_in-1))
        elif dp_constraint == "metric-l1" or dp_constraint == "metric-l2":
            data = np.absolute(np.arange(0, B_in)[:, None] - np.arange(0, B_in)[None, :]).reshape(B_in*B_in,) / (B_in-1)
            if dp_constraint == "metric-l2":
                data = np.power(data, 2)
            data = -np.exp(self.epsilon * data[data>0])
        else:
            raise RuntimeError("Unknown DP constraint: " + str(dp_constraint))
        row_indices = np.arange(0, B_in*(B_in-1)).astype(int)
        col_indices = np.floor(np.arange(0, B_in*(B_in-1)) / (B_in-1)).astype(int)
        coeffs = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(B_in*(B_in-1), B_in))
        D = sparse.kron(coeffs, sparse.eye(B_out))     # matrix of -e^{epsilons}s at all (i',j) positions
        E = sparse.csr_matrix(
            (np.ones(B_in), (np.arange(0, B_in*B_in, B_in+1).astype(int), np.arange(0, B_in).astype(int))), shape=(B_in*B_in, B_in))
        F = sparse.kron(np.ones(B_in)[:, None], sparse.eye(B_in*B_out)) - sparse.kron(E, sparse.eye(B_out))
        D += F[F.getnnz(1)>0]                      # add to D a matrix of 1s at all (i,j) positions
        return D
    
    def _get_row_stochastic_constraint_matrix(self):
        """
        Returns a sparse matrix R with shape (B, B*B) such that R @ p corresponds to the
        left-hand side of the row-stochastic constraints, where p = P.reshape(B*B).

        The final constraints are R @ p == 1.
        """
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        return sparse.kron(sparse.eye(B_in), np.ones(B_out))
    
    def _get_lp_costs(self, alpha, objective="squared"):
        """
        When fixing alpha and solving for P, the design problem is a linear program in the entries of P.
        This function returns the cost function values in a vectorized form.

        The overall MSE objective is $\sum_{i,j} P[i,j] * (i/(B-1) - alpha[j])^2$.
        This function returns a vector of length B*B with the coefficient $(i/(B-1) - alpha[j])^2$ in
        the corresponding position.
        """
        B_in = 2 ** self.input_bits
        target = np.arange(0, 1+1/B_in, 1/(B_in-1))
        if objective == "squared":
            c = np.power(target[:, None] - alpha[None, :], 2).flatten()
        elif objective == "absolute":
            c = np.absolute(target[:, None] - alpha[None, :]).flatten()
        else:
            raise RuntimeError("Unknown objective: " + str(objective))
        return c
    
    # ===================
    # Penalized LP method
    # ===================
    def _optimize_penalized_lp(self, objective="squared", dp_constraint="strict", lam=0, num_iters=1, verbose=False):
        """
        Estimate the optimal design by alternating between:
            a) Fixing alpha and solving for P, where the unbiased constraints are incorporated as a penalty in the objective,
            b) Fixing P and solving for alpha in terms of the linear system corresponding to unbiased constraints.
        """
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        alpha = np.arange(0, 1+1/B_out, 1/(B_out-1))
        target = np.arange(0, 1+1/B_in, 1/(B_in-1))
        num_var = B_in * B_out

        # Row-stochastic equality constraints: A_eq @ p == b_eq
        A_eq = self._get_row_stochastic_constraint_matrix()
        b_eq = np.ones(B_in)

        # DP inequality constraints: A_ineq @ p <= 0
        A_ineq = self._get_dp_constraint_matrix(dp_constraint)
        b_ineq = np.zeros(A_ineq.shape[0])
        P = None
        
        for l in range(num_iters):

            c = self._get_lp_costs(alpha, objective)

            # Coefficients to implement the unbiased constraint as a quadratic penalty
            beta = np.kron(target[:, None], alpha[None, :]).reshape(1, -1)[0]
            Q = np.kron(np.eye(B_in), np.kron(alpha[:, None], alpha[None, :]))
            Q += 1e-6 * np.eye(Q.shape[0])

            # Define and solve the CVXPY problem for P
            x = cvxpy.Variable(num_var, nonneg=True)
            if P is not None:
                x.value = P.flatten()
            if lam > 0:
                obj = cvxpy.quad_form(x, lam * Q) - 2 * lam * beta.T @ x + c.T @ x
            else:
                obj = c.T @ x
            prob = cvxpy.Problem(cvxpy.Minimize(obj), [A_ineq @ x <= b_ineq, A_eq @ x == b_eq])
            try:
                prob.solve(solver="ECOS", max_iters=1000, warm_start=(P is not None), verbose=verbose)
            except (cvxpy.error.SolverError, scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence):
                # Hack to continue even if solver fails to converge
                pass
            P = np.resize(x.value, (B_in, B_out))
            
            if lam > 0:
                # Define and solve the CVXPY problem for alpha
                x = cvxpy.Variable(B_out, nonneg=False)
                x.value = alpha
                obj = cvxpy.quad_form(x, lam * P.T @ P + np.diag(P.sum(0))) - 2 * (1 + lam) * (target @ P) @ x
                prob = cvxpy.Problem(cvxpy.Minimize(obj))
                prob.solve(solver="ECOS", max_iters=1000, warm_start=True, verbose=verbose)
                alpha = x.value
            else:
                # Obtain alpha by solving the linear system P @ alpha = target
                H = P.transpose().dot(P) + 1e-6 * np.eye(B_out)
                alpha = np.linalg.inv(H).dot(P.transpose().dot(target))
            
            mse_loss, bias_sq = self.mse_and_bias_squared(P, alpha)
            if verbose:
                print("Iteration %d: MSE loss = %.4f, squared bias = %.4f" % (
                    l+1, mse_loss, bias_sq))
            
        return P, alpha
    
    # ===========================================
    # Alternative method and supporting functions
    # ===========================================
    def _solve_lp_for_P(self, alpha, dp_constraint, verbose=False):
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        assert len(alpha) == B_out
        p = cvxpy.Variable(B_in * B_out, nonneg=True)
        u = np.arange(0, 1+1/B_in, 1/(B_in-1))
        # Cost matrix as a function of alpha
        c = self._get_lp_costs(alpha)
        objective = cvxpy.Minimize(c.T @ p)
        # DP constraints
        D = self._get_dp_constraint_matrix(dp_constraint)
        # Unbiased constraints
        A = sparse.kron(sparse.eye(B_in), alpha)
        # Row-stochastic constraint
        R = sparse.kron(sparse.eye(B_in), np.ones(B_out))
        # Formulate constraints
        constraints = [
            D @ p <= np.zeros(D.shape[0]),
            A @ p == u,
            R @ p == np.ones(B_in)
        ]
        # Build and solve the problem
        prob = cvxpy.Problem(objective, constraints)
        prob.solve()
        if verbose:
            print("Solving LP for P given alpha")
            print(f"Objective value is {prob.value}")
            if prob.value < math.inf:
                print(f"Max DP constraint violation is {np.max(constraints[0].violation())}")
                print(f"Max unbiased constraint violation is {np.max(constraints[1].violation())}")
                print(f"Max row-stochastic constraint violation is {np.max(constraints[2].violation())}")
        if prob.value == math.inf:
            p.value = np.zeros((B_in * B_out))
        return p.value.reshape((B_in, B_out)), prob.value

    def _solve_qp_for_alpha(self, P, verbose=False):
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        assert P.shape == (B_in, B_out)
        alpha = cvxpy.Variable(B_out)
        Q = np.diag(P.sum(axis=0))
        u = np.arange(0, 1+1/B_in, 1/(B_in-1))
        q = -2 * (u.T @ P)
        # Impose constraints on ordering (just to make things more easily interpretable)
        A = np.zeros((B_out-1, B_out))
        for i in range(B_out-1):
            A[i, i] = 1.0
            A[i, i+1] = -1.0
        objective = cvxpy.Minimize(cvxpy.quad_form(alpha, Q) + q.T @ alpha)
        constraints = [P @ alpha == u, A @ alpha <= np.zeros(B_out-1)]
        prob = cvxpy.Problem(objective, constraints)
        prob.solve()
        if verbose:
            print("Solving QP for alpha given P")
            print(f"Objective value is {prob.value}")
            if prob.value < math.inf:
                print(f"Max unbiased constraint violation is {np.max(constraints[0].violation())}")
                print(f"Max ordering constraint violation is {np.max(constraints[1].violation())}")
        return alpha.value, prob.value

    def _run_one_init(self, num_iters, verbose, alphainit, dp_constraint):
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        target = np.arange(0, 1+1/B_in, 1/(B_in-1))
        # Initialize a feasible alpha
        alpha = np.linspace(alphainit[0], alphainit[1], num=B_out, endpoint=True)
        for iter in range(num_iters):
            P, value = self._solve_lp_for_P(alpha, dp_constraint, verbose=verbose)
            if value < math.inf:
                alpha, value = self._solve_qp_for_alpha(P, verbose=verbose)
                mse_loss, bias_sq = self.mse_and_bias_squared(P, alpha)
            else:
                mse_loss = math.inf
                bias_sq = math.inf
            if verbose:
                print("Iteration %d: MSE loss = %.8f, squared bias = %.8f" % (
                    iter+1, mse_loss, bias_sq))
        return P, alpha, value
    
    def _optimize_alt(self, objective="squared", dp_constraint="strict", num_iters=1, verbose=False, alphainit=None, num_inits=10, Delta=1.0):
        if objective != "squared":
            raise RuntimeError("Unsupported objective: " + str(objective))
        
        best_P = None
        best_alpha = None
        best_mse_loss = math.inf
        if alphainit is None:
            # Binary search on initialization, starting from (-Delta, 1+Delta)
            delta = Delta
            delta_vals = [0.0]
            for num_tries in tqdm(range(num_inits), disable=(not verbose)):
                delta_vals.append(delta)
                alphainit = (0.0 - delta, 1.0 + delta)
                if verbose:
                    print(f"Trying with alphainit={alphainit}") if verbose else None
                P, alpha, value = self._run_one_init(num_iters, verbose, alphainit, dp_constraint)
                if value < math.inf:
                    mse_loss, bias_sq = self.mse_and_bias_squared(P, alpha)
                else:
                    mse_loss = math.inf
                if mse_loss < best_mse_loss:
                    best_mse_loss = mse_loss
                    best_P = P
                    best_alpha = alpha
                    np_delta_vals = np.array(delta_vals)
                    lower_delta = np.max(np_delta_vals[np_delta_vals < delta])
                    delta = (delta + lower_delta) / 2
                else:
                    if np.any(np.array(delta_vals) > delta):
                        np_delta_vals = np.array(delta_vals)
                        higher_delta = np.min(np_delta_vals[np_delta_vals > delta])
                        delta = (delta + higher_delta) / 2
                    else:
                        delta = 2.0 * delta
            P, alpha = best_P, best_alpha
        else:
            # Just run for the one value provided in alphainit
            P, alpha, value = self._run_one_init(num_iters, verbose, alphainit, dp_constraint)
            if value == math.inf:
                print(f"Did not find a feasible solution for alphainit={alphainit}")
        mse_loss, bias_sq = self.mse_and_bias_squared(P, alpha)
        if verbose:
            print("Final: MSE loss = %.8f, squared bias = %.8f" % (
                mse_loss, bias_sq))
        return P, alpha
    
    # ============================================
    # Trust-region method and supporting functions
    # ============================================
    def _get_objective(self):
        """
        Makes functions to compute the objective, gradient, and Hessian-vector product
        """
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        target = np.arange(0, 1+1/B_in, 1/(B_in-1))

        def objective(x):
            """
            Objective function to be minimized.
            x is a vectorized version of all optimization variables (entries
            of P reshaped as a vector, followed by entries of alpha).
            """
            P = x[:B_in*B_out].reshape((B_in, B_out))
            alpha = x[B_in*B_out:]
            return (P * np.power(target[:,None] - alpha[None,:], 2)).sum() / B_in
        
        def jac(x):
            """
            Gradient of the objective function at x wrt all parameters.
            """
            P = x[:B_in*B_out].reshape((B_in,B_out))
            alpha = x[B_in*B_out:]
            g = np.zeros(B_in*B_out + B_out)
            g[:B_in*B_out] = np.power(target[:, None] - alpha[None, :], 2).reshape(B_in * B_out)
            for j in range(B_out):
                g[B_in*B_out + j] = -2 * P[:,j].dot(np.arange(B_in)/(B_in-1) - alpha[j])
            return g
        
        def hessp(x, p):
            """
            Function that returns the product of p with the Hessian of the
            objective evaluated at x; i.e., H @ p.

            Note: Not currently used, but may be useful in the future.
            """
            P = x[:B_in*B_out].reshape((B_in,B_out))
            alpha = x[B_in*B_out:]
            # Hessian block for alpha-P cross terms, a B x (B*B) matrix with B*B non-zero elements
            row_ind = np.zeros(B_in * B_out)
            col_ind = np.zeros(B_in * B_out)
            data = np.zeros(B_in * B_out)
            next_nz = 0
            for j in range(B_out):
                for i in range(B_in):
                    row_ind[next_nz] = j
                    col_ind[next_nz] = i*B_out + j
                    data[next_nz] = -2*(i/(B_in-1) - alpha[j])
                    next_nz += 1
            hess_alpha_P = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(B_out, B_in*B_out))
            # Hessian block for alpha-alpha derivatives, a B x B matrix with B non-zero elements
            hess_alpha_alpha = sparse.diags(2 * P.sum(axis=0))
            Hp = np.zeros(B_in*B_out + B_out)
            Hp[:B_in*B_out] = hess_alpha_P.T @ p[B_in*B_out:]
            Hp[B_in*B_out:] = hess_alpha_P @ p[:B_in*B_out] + hess_alpha_alpha @ p[B_in*B_out:]
            return Hp

        def hess(x):
            """
            Function that returns the Hessian of the objective evaluated at x.
            """
            P = x[:B_in*B_out].reshape((B_in,B_out))
            alpha = x[B_in*B_out:]
            # Hessian block for alpha-P cross terms, a B x (B*B) matrix with B*B non-zero elements
            row_ind = np.zeros(B_in * B_out)
            col_ind = np.zeros(B_in * B_out)
            data = np.zeros(B_in * B_out)
            next_nz = 0
            for j in range(B_out):
                for i in range(B_in):
                    row_ind[next_nz] = j
                    col_ind[next_nz] = i*B_out + j
                    data[next_nz] = -2*(i/(B_in-1) - alpha[j])
                    next_nz += 1
            hess_alpha_P = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(B_out, B_in*B_out))
            # Hessian block for alpha-alpha derivatives, a B x B matrix with B non-zero elements
            hess_alpha_alpha = sparse.diags(2 * P.sum(axis=0))
            return sparse.bmat([[sparse.csr_matrix((B_in*B_out, B_in*B_out)), hess_alpha_P.T], [hess_alpha_P, hess_alpha_alpha]])
        
        return objective, jac, hess
    

    def _get_x0(self, verbose=False, dp_constraint="strict", init_method="random"):
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        if init_method == "alt":
            # Initialize the trust-region method from the solution returned by the Alternative solver.
            P, alpha = self._optimize_alt(self.budget, self.epsilon, dp_constraint=dp_constraint,
                                          verbose=verbose, num_inits=20, Delta=1.0)
            if verbose:
                print("Initializing from the Alternative solution")
                mse_loss, bias_sq = self.mse_and_bias_squared(P, alpha)
                print(f"Initial MSE is {mse_loss:.8f} and initial bias squared is {bias_sq:.8f}")
            x0 = np.hstack([P.reshape(B_in*B_out), alpha])
        elif init_method == "zeros":
            # Initialize from the all-zeros vector
            if verbose:
                print("Initializing from the all zeros vector")
            x0 = np.zeros(B_in*B_out + B_out)
        elif init_method == "random":
            # Random initialization
            if verbose:
                print("Initializing from a random vector")
            x0 = np.random.randn(B_in*B_out + B_out)
        elif init_method == "uniform":
            # Uniform initialization
            if verbose:
                print("Initializing with the uniform strategy")
            x0 = np.ones(B_in*B_out + B_out) / B_out
            x0[B_in*B_out:] = np.arange(B_out) / (B_out-1)
        else:
            raise RuntimeError("Unrecognized init_method passed to MVUMechanism with method=`trust-region`")
        return x0
    
    def _get_bounds(self):
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        ub = np.inf * np.ones(B_in*B_out + B_out)
        lb = np.zeros(B_in*B_out + B_out)
        lb[B_in*B_out:] = -np.inf
        return optimize.Bounds(lb, ub)
    
    def _get_dp_constraint(self, dp_constraint):
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        D = self._get_dp_constraint_matrix(dp_constraint)
        num_dp_constraints = D.shape[0]
        # This constraint gets applied to the full parameter vector;
        # pad with zeros to get the right shape
        Dext = sparse.hstack([D, sparse.csr_matrix((num_dp_constraints, B_out))])
        return optimize.LinearConstraint(Dext, -np.inf, 0)
    
    def _get_row_constraint(self):
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        R = self._get_row_stochastic_constraint_matrix()
        Rext = sparse.hstack([R, sparse.csr_matrix((B_in, B_out))])
        return optimize.LinearConstraint(Rext, 1, 1)

    def _get_unbiased_constraint(self):
        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        target = np.arange(0, 1+1/B_in, 1/(B_in-1))

        def unbiased_constraint_fn(x):
            P = x[:B_in*B_out].reshape((B_in,B_out))
            alpha = x[B_in*B_out:]
            return P @ alpha - target

        def unbiased_constraint_jac(x):
            P = x[:B_in*B_out].reshape((B_in,B_out))
            alpha = x[B_in*B_out:]
            # Return a B by (B*B + B) matrix with 2B * B non-zeros
            nnz = 2*B_in*B_out
            row_ind = np.zeros(nnz)
            col_ind = np.zeros(nnz)
            data = np.zeros(nnz)
            next_nz = 0
            for i in range(B_in):
                for j in range(B_out):
                    # \partial c_i / \partial P_{i,j}
                    row_ind[next_nz] = i
                    col_ind[next_nz] = i*B_out + j
                    data[next_nz] = alpha[j]
                    next_nz += 1
                    # \ partial c_i \partial alpha_j
                    row_ind[next_nz] = i
                    col_ind[next_nz] = B_in*B_out + j
                    data[next_nz] = P[i,j]
                    next_nz += 1
            return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(B_in, B_in*B_out + B_out))

        def unbiased_constraint_hess(x, v):
            P = x[:B_in*B_out].reshape((B_in,B_out))
            alpha = x[B_in*B_out:]
            # Compute alpha-P Hessian block, a B*B by B matrix with B*B non-zeros
            nnz = B_in*B_out
            row_ind = np.zeros(nnz)
            col_ind = np.zeros(nnz)
            data = np.zeros(nnz)
            next_nz = 0
            for i in range(B_in):
                for j in range(B_out):
                    # Entry corresponding to \partial^2 / (\partial P_{i,j} \partial \alpha_j)
                    row_ind[next_nz] = i*B_out + j
                    col_ind[next_nz] = j
                    data[next_nz] = v[i]
                    next_nz += 1
            hess_block = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(B_in*B_out, B_out))
            return sparse.bmat([[sparse.csr_matrix((B_in*B_out, B_in*B_out)), hess_block], [hess_block.T, sparse.csr_matrix((B_out, B_out))]])

        unbiased_constraint = optimize.NonlinearConstraint(
            unbiased_constraint_fn, np.zeros(B_in), np.zeros(B_in),
            unbiased_constraint_jac,
            unbiased_constraint_hess,
        )
        return unbiased_constraint
    
    def _get_constraints(self, dp_constraint):
        dp_constraint = self._get_dp_constraint(dp_constraint)
        row_constraint = self._get_row_constraint()
        unbiased_constraint = self._get_unbiased_constraint()
        return [dp_constraint, row_constraint, unbiased_constraint]

    def _optimize_tr(self, objective="squared", dp_constraint="strict", maxiter=5000, verbose=False, init_method="random"):
        if objective != "squared":
            raise RuntimeError("Unsupported objective: " + str(objective))

        if verbose:
            verbose_level = 3
        else:
            verbose_level = 0

        # objective, jac, hessp = self._get_objective()
        objective, jac, hess = self._get_objective()
        x0 = self._get_x0(verbose=verbose, dp_constraint=dp_constraint, init_method=init_method)
        bounds = self._get_bounds()
        constraints = self._get_constraints(dp_constraint)
        self.log = TRLoggerCallback(x0, self.budget, self.epsilon, self.input_bits, init_method, dp_constraint)

        result = optimize.minimize(
            objective, x0, method="trust-constr",
            jac=jac, hess=hess, bounds=bounds,
            constraints=constraints, callback=self.log,
            options={"verbose": verbose_level, "maxiter": maxiter, "sparse_jacobian": True},
        )
        if verbose:
            if result.success:
                print(f"Solver succeeded! {result.message}")
            else:
                # Note: Even when the solver does not succeed, it doesn't mean
                # that the solution is necessarily bad.
                print(f"Warning: Solver did not succeed. {result.message}")

        B_in = 2 ** self.input_bits
        B_out = 2 ** self.budget
        P = result.x[:B_in*B_out].reshape((B_in,B_out))
        alpha = result.x[B_in*B_out:]
        # Sort alpha values in ascending order
        perm = np.argsort(alpha)
        alpha = alpha[perm]
        P = P[:,perm]
        if verbose:
            mse_loss, bias_sq = self.mse_and_bias_squared(P, alpha)
            print("Final: MSE loss = %.8f, squared bias = %.8f" % (
                mse_loss, bias_sq))
        return P, alpha

    # =================================================================
    # Master function that dispatches to method-specific implementation
    # =================================================================
    def optimize(self, method, **kwargs):
        if method == "penalized-lp":
            return self._optimize_penalized_lp(**kwargs)
        elif method == "alt":
            return self._optimize_alt(**kwargs)
        elif method == "trust-region":
            return self._optimize_tr(**kwargs)
        else:
            raise RuntimeError(f"Unrecognized method `{method}`. Valid options are penalized-lp, alt, and trust-region.")


class TRLoggerCallback:
    """
    Helper class used in the trust-region solver.
    Logs some metrics at each step of the trust-region method, which may be useful
    for tuning the solver and/or debugging.
    """

    def __init__(self, x0, budget, epsilon, input_bits, init_method, dp_constraint):
        self.now = datetime.now()
        self.budget = budget
        self.input_bits = input_bits
        self.B_in = 2 ** self.input_bits
        self.B_out = 2 ** self.budget
        self.epsilon = epsilon
        self.init_method = init_method
        self.dp_constraint = dp_constraint
        self.columns = [
            'optimality', # Infinity norm of the Lagrangian gradient
            'constr_violation', # Maximum constraint violation
            'fun', # Function value
            'tr_radius', # Trust region radius
            'constr_penalty', # Constraint penalty parameter
            'barrier_tolerance', # Tolerance for barrier subproblem
            'barrier_parameter', # Barrier parameter
            'execution_time', # Total execution time
        ]
        self.results = {}
        for col in self.columns:
            self.results[col] = []
        self.results['x_diff_norm'] = []
        self.results['P_diff_l1_norm'] = []
        self.results['alpha_diff_l1_norm'] = []
        self.x_prev = x0.copy()
        self.results['dist_from_init'] = []
        self.x0 = x0.copy()
    
    def __call__(self, x, result):
        for col in self.columns:
            self.results[col].append(getattr(result, col))
        self.results['x_diff_norm'].append(np.linalg.norm(self.x_prev - result.x))
        self.results['P_diff_l1_norm'].append(np.linalg.norm(self.x_prev[:self.B_in*self.B_out] - result.x[:self.B_in*self.B_out], 1))
        self.results['alpha_diff_l1_norm'].append(np.linalg.norm(self.x_prev[self.B_in*self.B_out:] - result.x[self.B_in*self.B_out:], 1))
        np.copyto(self.x_prev, result.x)
        self.results['dist_from_init'].append(np.linalg.norm(self.x0 - result.x))
        return False


def save_mechanism(mechanism, path):
    """
    Save a mechanism to disk.
    """
    with open(path, 'wb') as f:
        pickle.dump(mechanism, f)


def load_mechanism(path):
    """
    Load a mechanism from disk.
    """
    with open(path, 'rb') as f:
        mechanism = pickle.load(f)
    return mechanism
