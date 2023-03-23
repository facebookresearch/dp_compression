#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import os
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from opacus.validators import ModuleValidator
from mechanisms import *
from mechanisms_pytorch import *
from utils import renyi_div_bound_lp, max_divergence_bound, fisher_information_bound, consolidate, params_to_vec, set_grad_to_vec

import sys
sys.path.append("Handcrafted-DP/")
from data import get_scatter_transform, get_scattered_loader
from models import ScatterLinear
from tqdm import trange


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, padding=0)
        self.fc1 = nn.Linear(32 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, 1)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

    
def clip_gradient(args, grad_vec, p=2):
    """
    L2 norm clip to args.norm_clip and then L-inf norm clip to args.linf_clip.
    """
    C = args.norm_clip
    grad_norm = grad_vec.norm(p, 1)
    multiplier = grad_norm.new(grad_norm.size()).fill_(1)
    multiplier[grad_norm.gt(C)] = C / grad_norm[grad_norm.gt(C)]
    grad_vec *= multiplier.unsqueeze(1)
    grad_vec.clamp_(-args.linf_clip, args.linf_clip)
    return grad_vec


def add_noise(args, grad_vec, device, mechanism="gaussian"):
    """
    Add noise and quantize the output if args.quantization > 0.
    """
    batch_size = grad_vec.size(0)
    d = grad_vec.size(1)
    if mechanism == "laplace":
        dist = torch.distributions.laplace.Laplace(0, 1)
        grad_vec += dist.sample(grad_vec.size()).to(device) * args.norm_clip * args.scale
        if args.quantization > 0:
            assert args.quantization == 1, "Laplace mechanism with quantization level > 1 is not implemented yet."
            grad_vec = grad_vec.sign()
    elif mechanism == "gaussian":
        grad_vec += torch.randn_like(grad_vec).to(device) * args.norm_clip * args.scale
        if args.quantization > 0:
            assert args.quantization == 1, "Gaussian mechanism with quantization level > 1 is not implemented yet."
            grad_vec = grad_vec.sign()
    elif isinstance(mechanism, SkellamMechanismPyTorch):
        grad_vec = mechanism.decode(mechanism.privatize(mechanism.scale * grad_vec))
    elif isinstance(mechanism, MVUMechanismPyTorch) or isinstance(mechanism, IMVUMechanismPyTorch):
        M = args.linf_clip
        # scale input to [0,1]
        normalized_grad_vec = ((mechanism.scale * grad_vec + M) / (2 * M)).clamp(0, 1)
        privatized_grad_vec = mechanism.decode(mechanism.privatize(normalized_grad_vec))
        # scale input back to [-M, M]
        grad_vec = privatized_grad_vec.float().view(batch_size, -1) * 2 * M - M
        grad_vec /= mechanism.scale
    else:
        raise NotImplementedError(mechanism)
    return grad_vec.sum(0)


def train(args, model, device, train_loader, optimizer, epoch, mechanism="gaussian"):
    num_param = sum([np.prod(layer.size()) for layer in model.parameters()])
    model.train()
    p = 1 if args.mechanism == "laplace" or args.mechanism == "mvu_l1" else 2
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        num_batches = int(math.ceil(float(data.size(0)) / args.physical_batch_size))
        grad_sum = torch.zeros(num_param).to(device)
        for i in range(num_batches):
            model.zero_grad()
            lower = i * args.physical_batch_size
            upper = min((i+1) * args.physical_batch_size, data.size(0))
            x, y = data[lower:upper], target[lower:upper]
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad_vec = params_to_vec(model, return_type="grad_sample")
            d = grad_vec.size(1)
            if args.norm_clip > 0:
                grad_vec = clip_gradient(args, grad_vec, p)
                grad_sum += add_noise(args, grad_vec, device, mechanism)
            else:
                grad_sum += grad_vec.sum(0)
        grad_mean = grad_sum / data.size(0)
        set_grad_to_vec(model, grad_mean)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return float(correct) / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description='DP-SGD MNIST and CIFAR10 training')
    parser.add_argument('--save-dir', type=str, default='dpsgd_results',
                        help='save directory')
    parser.add_argument('--batch-size', type=int, default=600,
                        help='(virtual) input batch size for training')
    parser.add_argument('--physical-batch-size', type=int, default=0,
                        help='actual batch size')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing')
    parser.add_argument('--dataset', type=str, default='mnist', choices=["mnist", "fmnist", "kmnist", "cifar10"],
                        help='which dataset to train on')
    parser.add_argument('--model', type=str, choices=['linear', 'convnet'], default='convnet',
                        help='which model to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum')
    parser.add_argument('--norm-clip', type=float, default=0,
                        help='gradient norm clip')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta scaling for MVU; must be >0')
    parser.add_argument('--mechanism', type=str, default='gaussian',
                        choices=["laplace", "gaussian", "mvu", "mvu_l1", "mvu_l2", "skellam"],
                        help='which mechanism to use')
    parser.add_argument('--quantization', type=int, default=0,
                        help='quantization level for linf clipping')
    parser.add_argument('--input-bits', type=int, default=1,
                        help='number of input bits for MVU mechanism')
    parser.add_argument('--epsilon', type=float, default=1,
                        help='DP epsilon for MVU mechanism')
    parser.add_argument('--scale', type=float, default=0,
                        help='Laplace/Gaussian noise multiplier')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for saving the current model')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.linf_clip = args.norm_clip / args.beta
    if args.physical_batch_size == 0:
        args.physical_batch_size = args.batch_size

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
    
    if args.mechanism.startswith("mvu"):
        output_file = "%s/%s_%s_epochs_%d_lr_%.2e_clip_%.2e_beta_%.2e_%s_bin_%d_quant_%d_eps_%.2e.pth" % (
            args.save_dir, args.dataset, args.model, args.epochs, args.lr, args.norm_clip, args.beta,
            args.mechanism, args.input_bits, args.quantization, args.epsilon
        )
    else:
        output_file = "%s/%s_%s_epochs_%d_lr_%.2e_clip_%.2e_beta_%.2e_%s_quant_%d_scale_%.2e.pth" % (
            args.save_dir, args.dataset, args.model, args.epochs, args.lr, args.norm_clip, args.beta,
            args.mechanism, args.quantization, args.scale
        )

    ### DATA LOADING ###
    transform = transforms.ToTensor()
    if args.dataset == 'mnist':
        train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST('./data', train=False, transform=transform)
    elif args.dataset == 'fmnist':
        train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST('./data', train=False, transform=transform)
    elif args.dataset == 'kmnist':
        train_set = datasets.KMNIST('./data', train=True, download=True, transform=transform)
        test_set = datasets.KMNIST('./data', train=False, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform)
        
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    ### MODEL LOADING ###
    if args.model == "linear":
        scattering, K, (h, w) = get_scatter_transform("cifar10" if args.dataset == "cifar10" else "mnist")
        scattering.to(device)
        train_loader = get_scattered_loader(train_loader, scattering, device, drop_last=True, sample_batches=False)
        test_loader = get_scattered_loader(test_loader, scattering, device)
        model = GradSampleModule(ScatterLinear(K, (h, w), input_norm="GroupNorm", num_groups=27).to(device))
    else:
        assert args.dataset != "cifar10", "CIFAR-10 ConvNet training is not supported."
        model = GradSampleModule(ConvNet().to(device))
    num_param = sum([np.prod(layer.size()) for layer in model.parameters()])
    print("Number of model parameters = %d" % num_param)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    ### MECHANISM LOADING ###
    if args.mechanism.startswith("mvu") and args.norm_clip > 0 and args.linf_clip > 0:
        savefile = os.path.join('sweep_eps_budget_penalized_lam1.0e+02', f"mechanism_bin{args.input_bits}_bout{args.quantization}_metric-l1_eps{args.epsilon:.2f}.pkl")
        with open(savefile, "rb") as file:
            mechanism_numpy = pickle.load(file)
        mechanism_numpy.P /= mechanism_numpy.P.sum(1)[:, None]
        norm_bound = args.norm_clip / (2 * args.linf_clip)    # L2 sensitivity for leave-one-out adjacency
        if args.mechanism == "mvu":
            mechanism = MVUMechanismPyTorch(
                args.input_bits, args.quantization, args.epsilon, torch.from_numpy(mechanism_numpy.P),
                torch.from_numpy(mechanism_numpy.alpha), norm_bound, device)
            mechanism.scale = 0.9
            print("Computing Renyi divergence bounds using LP relaxation")
            renyi_div_bounds = renyi_div_bound_lp(orders, num_param, mechanism_numpy.P, norm_bound, greedy=True)
        elif args.mechanism == "mvu_l1":
            mechanism = IMVUMechanismPyTorch(
                args.input_bits, args.quantization, torch.from_numpy(mechanism_numpy.P),
                torch.from_numpy(mechanism_numpy.alpha), device)
            log_P = np.log(mechanism_numpy.P)
            epsilon_bound = args.epsilon + max_divergence_bound(log_P)
            print("Max divergence bound = %.4f" % epsilon_bound)
        else:
            assert args.input_bits == 1, "Interpolated MVU is not defined for b_in > 1"
            mechanism_numpy.P[1, :] = np.flip(mechanism_numpy.P[0, :], (0,))
            mechanism = IMVUMechanismPyTorch(
                args.input_bits, args.quantization, torch.from_numpy(mechanism_numpy.P),
                torch.from_numpy(mechanism_numpy.alpha), device)
            P, _ = consolidate(mechanism_numpy)
            fisher_info_bound = fisher_information_bound(P[0, :])
            print("Fisher info bound = %.4f" % fisher_info_bound)
    elif args.mechanism == "skellam":
        mu = (args.scale * args.norm_clip)**2
        mechanism = SkellamMechanismPyTorch(args.quantization, num_param, args.norm_clip, mu, device)
    else:
        mechanism = args.mechanism
        
    q = args.batch_size / float(len(train_set))
    orders = np.array(list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64)))
    test_accs, epsilons = torch.zeros(args.epochs), torch.zeros(args.epochs)
    for epoch in range(1, args.epochs + 1):
        ### TRAINING ###
        train(args, model, device, train_loader, optimizer, epoch, mechanism)
        test_accs[epoch-1] = test(model, device, test_loader)
        ### PRIVACY ACCOUNTING ###
        delta = 1e-5
        if args.mechanism == "laplace" and args.scale > 0 and args.norm_clip > 0:
            delta = 0
            opt_order = 0
            epsilon = epoch / args.scale
        elif args.mechanism == "gaussian" and args.scale > 0 and args.norm_clip > 0:
            rdp_const = epoch * orders / (2 * args.scale ** 2)
            epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp_const, delta=delta)
        elif args.mechanism == "mvu" and args.norm_clip > 0:
            rdp_const = renyi_div_bounds * epoch
            epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp_const, delta=delta)
        elif args.mechanism == "mvu_l1" and args.norm_clip > 0:
            delta = 0
            opt_order = 0
            epsilon = epoch * epsilon_bound * norm_bound
        elif args.mechanism == "mvu_l2" and args.norm_clip > 0:
            rdp_const = epoch * orders * fisher_info_bound * norm_bound**2 / 2
            epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp_const, delta=delta)
        elif args.mechanism == "skellam" and args.norm_clip > 0:
            rdp_const = epoch * mechanism.renyi_div(orders)
            epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp_const, delta=delta)
        else:
            epsilon, opt_order = math.inf, 0
        epsilons[epoch-1] = epsilon
        print("Epsilon at delta=%.2e: %.4f, optimal alpha: %.4f\n" % (delta, epsilon, opt_order))

    if args.save_model:
        if os.path.exists(output_file):
            checkpoint = torch.load(output_file)
            checkpoint['state_dict'].append(model.state_dict())
            checkpoint['test_accs'].append(test_accs)
            checkpoint['epsilons'].append(epsilons)
            torch.save(checkpoint, output_file)
        else:
            torch.save({'state_dict': [model.state_dict()], 'test_accs': [test_accs], 'epsilons': [epsilons]}, output_file)


if __name__ == '__main__':
    main()
