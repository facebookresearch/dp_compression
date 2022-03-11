#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from opacus.accountants.analysis.rdp import get_privacy_spent

epsilons = np.arange(0.5, 5.1, 0.5)
epsilons_approx = np.zeros(epsilons.shape)
mean_cldp, std_cldp = np.zeros(epsilons.shape), np.zeros(epsilons.shape)
mean_skellam, std_skellam = np.zeros(epsilons.shape), np.zeros(epsilons.shape)
mean_mvu, std_mvu = np.zeros(epsilons.shape), np.zeros(epsilons.shape)
mean_mvu_approx, std_mvu_approx = np.zeros(epsilons_approx.shape), np.zeros(epsilons_approx.shape)
mean_baseline, std_baseline = np.zeros(epsilons.shape), np.zeros(epsilons.shape)

num_samples = 10000
for i in range(len(epsilons)):
    checkpoint = np.load("dme_results/dme_multi_l2_d_128_samples_%d_eps_%.2f_skellam_16_15.00_mvu_3_5.npz" % (num_samples, epsilons[i]))
    mean_cldp[i] = checkpoint["squared_error_cldp"].mean()
    std_cldp[i] = checkpoint["squared_error_cldp"].std()
    mean_skellam[i] = checkpoint["squared_error_skellam"].mean()
    std_skellam[i] = checkpoint["squared_error_skellam"].std()
    mean_mvu[i] = checkpoint["squared_error_mvu"].mean()
    std_mvu[i] = checkpoint["squared_error_mvu"].std()
    mean_mvu_approx[i] = checkpoint["squared_error_mvu_approx"].mean()
    std_mvu_approx[i] = checkpoint["squared_error_mvu_approx"].std()
    # use computed metric L2 renyi divergence bounds
    renyi_divs = np.load("sweep_eps_budget_penalized_lam1.0e+02/renyi_div_bin5_bout3_metric-l1_eps%.2f_Delta0.50.npz" % (0.5 * epsilons[i]))
    epsilons_approx[i] = get_privacy_spent(orders=renyi_divs["alphas"], rdp=renyi_divs["renyi_div_bound"],
                                            delta=(1/(num_samples+1)))[0]
    mean_baseline[i] = checkpoint["squared_error_baseline"].mean()
    std_baseline[i] = checkpoint["squared_error_baseline"].std()
    
plt.figure(figsize=(8,6))
with sns.color_palette("deep"):
    plt.errorbar(epsilons, mean_cldp, std_cldp, label="CLDP", linewidth=3)
    plt.errorbar(epsilons, mean_skellam, std_skellam, label="Skellam ($\\delta > 0$)", linewidth=3)
    plt.errorbar(epsilons, mean_mvu, std_mvu, label="MVU", linewidth=3)
plt.errorbar(epsilons_approx, mean_mvu_approx, std_mvu_approx, label="MVU ($\\delta > 0$)", linewidth=3, color="lightgreen")
plt.errorbar(epsilons, mean_baseline, std_baseline, label="Gaussian ($\\delta > 0$)", color='k', linestyle='--', linewidth=3)

plt.yscale('log')
plt.xticks(fontsize=20)
plt.xlabel("$\\epsilon$", fontsize=24)
plt.yticks(fontsize=20)
plt.ylabel("MSE", fontsize=24)
plt.legend(fontsize=20, loc="upper right")
plt.grid('on')
plt.title("$L_2, d=128$", fontsize=24)
plt.savefig("figures/dme_l2.pdf", bbox_inches="tight")
