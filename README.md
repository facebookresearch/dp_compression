# Privacy-Aware Compression for Federated Data Analysis

This repository contains code for reproducing results in the papers:
- Kamalika Chaudhuri*, Chuan Guo*, Mike Rabbat. **[Privacy-Aware Compression for Federated Data Analysis](https://arxiv.org/abs/2203.08134)**.
- Chuan Guo, Kamalika Chaudhuri, Pierre Stock, Mike Rabbat. **[Privacy-Aware Compression for Federated Learning Through Numerical Mechanism Design](https://arxiv.org/abs/2211.03942)**.

## Setup

Dependencies: [numpy](https://numpy.org/), [scipy](https://scipy.org/), [cvxpy](https://www.cvxpy.org/), [pytorch](https://pytorch.org/), [opacus](https://github.com/pytorch/opacus), [kymatio](https://github.com/kymatio/kymatio), [Handcrafted-DP](https://github.com/ftramer/Handcrafted-DP), [private_prediction](https://github.com/facebookresearch/private_prediction), [fastwht](https://github.com/vegarant/fastwht).

After cloning repo and installing dependencies (see `requirements.txt`), download submodules and run the install script to apply some patches.
```
git submodule update --init
python install.py
cd fastwht/python
./setup.sh
```

## Experiments

### Scalar Distributed Mean Estimation (CPU only)

```
for epsilon in 1 3 5; do
    python optimize_mvu.py --input_bits 3 --budget 3 --epsilon $epsilon --dp_constraint strict --method tr
    python mean_estimation_single.py --epsilon $epsilon
done
```

### Vector Distributed Mean Estimation (CPU only)

For L1-sensitivity setting, first optimize the MVU mechanisms:
```
for epsilon in 1 2 3 4 5 6 7 8 9 10; do
    python optimize_mvu.py --input_bits 9 --budget 3 --epsilon $epsilon --dp_constraint metric-l1 --method penalized
done
```
Then run the DME experiment and plot result:
```
for epsilon in 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5; do
    python mean_estimation_multi.py --norm_type l1 --epsilon $epsilon --skellam_budget 16 --skellam_s 100 --mvu_input_bits 9 --mvu_budget 3
done
python plot_dme_l1.py
```

For L2-sensitivity setting, first optimize the MVU mechanisms and compute Renyi divergence curve for both the pure and approximate DP variants:
```
for epsilon in 2 4 6 8 10 12 14 16 18 20; do
    python optimize_mvu.py --input_bits 5 --budget 3 --epsilon $epsilon --dp_constraint metric-l2 --method penalized
done
for epsilon in 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5; do
    python optimize_mvu.py --input_bits 5 --budget 3 --epsilon $epsilon --dp_constraint metric-l1 --method penalized
    python compute_renyi_div.py --input_bits 5 --budget 3 --epsilon $epsilon --dp_constraint metric-l1
done
```
Then run the DME experiment and plot result:
```
for epsilon in 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5; do
    python mean_estimation_multi.py --norm_type l2 --epsilon $epsilon --skellam_budget 16 --skellam_s 15 --mvu_input_bits 5 --mvu_budget 3
done
python plot_dme_l2.py
```

**Note**: These experiments will take a few hours to run.

### DP-SGD Training (requires GPU)


To run the DP-SGD training experiment, first optimize the MVU mechanism:
```
python optimize_mvu.py --input_bits 9 --budget 1 --epsilon <epsilon> --dp_constraint metric-l1 --method penalized
python optimize_mvu.py --input_bits 1 --budget 1 --epsilon <epsilon> --dp_constraint metric-l1 --method penalized
```
Then run DP-SGD training with Gaussian mechanism, signSGD, Skellam, MVU and I-MVU:
```
python train.py --save-model --dataset mnist --model <convnet/linear> --mechanism gaussian --quantization 0 --epochs <epochs> --scale <sigma> --lr <lr> --norm-clip <norm_clip>
python train.py --save-model --dataset mnist --model <convnet/linear> --mechanism gaussian --quantization 1 --epochs <epochs> --scale <sigma> --lr <lr> --norm-clip <norm_clip>
python train.py --save-model --dataset mnist --model <convnet/linear> --mechanism skellam --quantization 16 --epochs <epochs> --scale <sigma> --lr <lr> --norm-clip <norm_clip>
python train.py --save-model --dataset mnist --model <convnet/linear> --mechanism mvu --input-bits 9 --quantization 1 --beta 1 --epochs <epochs> --epsilon <epsilon> --lr <lr> --norm-clip <norm_clip>
python train.py --save-model --dataset mnist --model <convnet/linear> --mechanism mvu_l2 --input-bits 1 --quantization 1 --beta 1 --epochs <epochs> --epsilon <epsilon> --lr <lr> --norm-clip <norm_clip>
```
To train on CIFAR-10, simply replace `--dataset mnist` by `--dataset cifar10`. See appendix in our paper for the full grid of hyperparameter values.

## Code Acknowledgements

The majority of Privacy-Aware Compression is licensed under CC-BY-NC, however portions of the project are available under separate license terms: CVXPY and Opacus are licensed under the Apache 2.0 license; Kymatio is licensed under the BSD license; and Handcrafted-DP is licensed under the MIT license.
