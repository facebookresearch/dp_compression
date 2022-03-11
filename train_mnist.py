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
from torchvision import datasets, transforms
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from mechanisms import *
from utils import renyi_div_bound_lp, optimal_scaling_mvu

import sys
sys.path.append("Handcrafted-DP/")
from data import get_scatter_transform, get_scattered_loader
from models import ScatterLinear


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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=2)
        self.conv2 = nn.Conv2d(16, 64, 4, 2, padding=0)
        self.fc1 = nn.Linear(64 * 5 * 5, 32)
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
    
    
class MVUMechanismPyTorch:
    
    def __init__(self, budget, epsilon, input_bits, P, alpha, norm_bound, device):
        self.budget = budget
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
    

def clip_gradient(args, grad_vec):
    """
    L2 norm clip to args.norm_clip and then L-inf norm clip to args.linf_clip.
    """
    C = args.norm_clip
    grad_norm = grad_vec.norm(2, 1)
    multiplier = grad_norm.new(grad_norm.size()).fill_(1)
    multiplier[grad_norm.gt(C)] = C / grad_norm[grad_norm.gt(C)]
    grad_vec *= multiplier.unsqueeze(1)
    if args.linf_clip > 0:
        grad_vec.clamp_(-args.norm_clip, args.norm_clip)
    return grad_vec


def add_noise(args, grad_vec, device, mechanism=None):
    """
    Add noise and quantize the output if args.quantization > 0.
    """
    batch_size = grad_vec.size(0)
    d = grad_vec.size(1)
    if mechanism is None:
        # default to Gaussian mechanism
        grad_vec += torch.randn_like(grad_vec).to(device) * args.norm_clip * args.sigma
        if args.quantization > 0:
            assert args.quantization == 1, "Gaussian mechanism with quantization level > 1 is not implemented yet."
            grad_vec = grad_vec.sign()
    else:
        M = args.linf_clip
        # scale input to [0,1]
        normalized_grad_vec = ((mechanism.scale * grad_vec + M) / (2 * M)).clamp(0, 1)
        privatized_grad_vec = mechanism.decode(mechanism.privatize(normalized_grad_vec))
        # scale input back to [-M, M]
        grad_vec = privatized_grad_vec.float().view(batch_size, -1) * 2 * M - M
        grad_vec /= mechanism.scale
    return grad_vec.mean(0)


def train(args, model, device, train_loader, optimizer, epoch, mechanism=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        grad_vec = params_to_vec(model, return_type="grad_sample")
        if args.norm_clip > 0:
            grad_vec = clip_gradient(args, grad_vec)
            grad_mean = add_noise(args, grad_vec, device, mechanism)
        else:
            grad_mean = grad_vec.mean(0)
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
    parser = argparse.ArgumentParser(description='DP-SGD MNIST training')
    parser.add_argument('--save-dir', type=str, default='dpsgd_results',
                        help='save directory')
    parser.add_argument('--batch-size', type=int, default=600,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing')
    parser.add_argument('--dataset', type=str, default='mnist', choices=["mnist", "fmnist", "kmnist"],
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
    parser.add_argument('--linf-multiplier', type=float, default=0,
                        help='L-inf clipping multiplier, must be non-zero when using MVU mechanism')
    parser.add_argument('--mechanism', type=str, default='gaussian', choices=["gaussian", "mvu"],
                        help='which mechanism to use')
    parser.add_argument('--quantization', type=int, default=0,
                        help='quantization level for linf clipping')
    parser.add_argument('--input-bits', type=int, default=9,
                        help='number of input bits for MVU mechanism')
    parser.add_argument('--epsilon', type=float, default=1,
                        help='DP epsilon for MVU mechanism')
    parser.add_argument('--sigma', type=float, default=0,
                        help='Gaussian noise multiplier')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for saving the current model')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # args.linf_multiplier is the ratio M / C <= 1
    args.linf_clip = args.linf_multiplier * args.norm_clip

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
    
    if args.mechanism == "gaussian":
        output_file = "%s/%s_%s_epochs_%d_lr_%.2e_clip_%.2e_linf_%.2e_quant_%d_sigma_%.2e.pth" % (
            args.save_dir, args.dataset, args.model, args.epochs, args.lr, args.norm_clip, args.linf_multiplier,
            args.quantization, args.sigma
        )
    else:
        output_file = "%s/%s_%s_epochs_%d_lr_%.2e_clip_%.2e_linf_%.2e_quant_%d_eps_%.2e.pth" % (
            args.save_dir, args.dataset, args.model, args.epochs, args.lr, args.norm_clip, args.linf_multiplier,
            args.quantization, args.epsilon
        )
    if os.path.exists(output_file) and args.save_model:
        print('Result already exists; skipping.')
        exit()

    transform = transforms.ToTensor()
    if args.dataset == 'fmnist':
        train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST('./data', train=False, transform=transform)
    elif args.dataset == 'kmnist':
        train_set = datasets.KMNIST('./data', train=True, download=True, transform=transform)
        test_set = datasets.KMNIST('./data', train=False, transform=transform)
    else:
        train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST('./data', train=False, transform=transform)
        
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    if args.model == "linear":
        scattering, K, (h, w) = get_scatter_transform("mnist")
        scattering.to(device)
        train_loader = get_scattered_loader(train_loader, scattering, device, drop_last=True, sample_batches=False)
        test_loader = get_scattered_loader(test_loader, scattering, device)
        model = GradSampleModule(ScatterLinear(K, (h, w), input_norm="GroupNorm", num_groups=24).to(device))
    else:
        model = GradSampleModule(ConvNet().to(device))
    num_param = sum([np.prod(layer.size()) for layer in model.parameters()])
    print("Number of model parameters = %d" % num_param)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    test_accs, epsilons = torch.zeros(args.epochs), torch.zeros(args.epochs)
    
    q = args.batch_size / float(len(train_set))
    orders = np.array(list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64)))
    if args.mechanism == "mvu" and args.norm_clip > 0 and args.linf_clip > 0:
        assert args.linf_clip > 0, "MVU mechanism requires L-inf clipping."
        savefile = os.path.join('sweep_eps_budget_penalized_lam1.0e+02', f"mechanism_bin{args.input_bits}_bout{args.quantization}_metric-l1_eps{args.epsilon:.2f}.pkl")
        with open(savefile, "rb") as file:
            mechanism_numpy = pickle.load(file)
        mechanism_numpy.P /= mechanism_numpy.P.sum(1)[:, None]
        mechanism = MVUMechanismPyTorch(
            args.quantization, args.epsilon, args.input_bits, torch.from_numpy(mechanism_numpy.P),
            torch.from_numpy(mechanism_numpy.alpha), args.norm_clip / (2 * args.linf_clip), device)
        # generate samples to estimate optimal scaling factor
        print("Computing optimal scaling factor")
        samples = np.random.normal(0, 1, (3000, num_param))
        samples /= np.linalg.norm(samples, 2, 1)[:, None]
        mechanism.scale = 0.95 * optimal_scaling_mvu(samples, mechanism_numpy, conf=0.01)
        print("Computing Renyi divergence bounds using LP relaxation")
        renyi_div_bounds = renyi_div_bound_lp(orders, num_param, mechanism_numpy.P,
                                              args.norm_clip / (2 * args.linf_clip), greedy=True)
    else:
        mechanism = None
        
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, mechanism)
        test_accs[epoch-1] = test(model, device, test_loader)
        ### PRIVACY ACCOUNTING
        if args.mechanism == "gaussian" and args.sigma > 0 and args.norm_clip > 0:
            rdp_const = epoch * orders / (2 * args.sigma ** 2)
            epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp_const, delta=1e-5)
        elif args.mechanism == "mvu" and args.norm_clip > 0 and args.linf_clip > 0:
            rdp_const = renyi_div_bounds * epoch
            epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp_const, delta=1e-5)
        else:
            epsilon, opt_order = math.inf, 0
        epsilons[epoch-1] = epsilon
        print("Epsilon at delta=1e-5: %.4f, optimal alpha: %.4f\n" % (epsilon, opt_order))

    if args.save_model:
        torch.save({'state_dict': model.state_dict(), 'test_accs': test_accs, 'epsilons': epsilons}, output_file)


if __name__ == '__main__':
    main()