# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 01:30:24 2020

@author: Jake
"""

import time
import torch
import numpy as np
import random
from torch.autograd import grad


def hessian_vector_product(ys, xs, v):
    J = grad(ys, xs, create_graph=True)[0]
    grads = grad(J, xs, v, retain_graph=True)
    del J, ys, v
    torch.cuda.empty_cache()
    return grads


def lissa(train_loss, test_loss, layer_weight, model):
    scale = 10
    damping = 0.1
    num_samples = 1
    v = grad(test_loss, layer_weight)[0]
    cur_estimate = v.clone()
    prev_norm = 1
    diff = prev_norm
    count = 0
    while diff > 0.00001:
        hvp = hessian_vector_product(train_loss, layer_weight, cur_estimate)
        cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, cur_estimate, hvp)]
        cur_estimate = torch.squeeze(torch.stack(cur_estimate))
        model.zero_grad()
        numpy_est = cur_estimate.detach().cpu().numpy()
        numpy_est = numpy_est.reshape(1, -1)

        if (count % 100 == 0):
            print("Recursion at depth %s: norm is %.8lf" % (count, np.linalg.norm(np.concatenate(numpy_est))))
        count += 1
        diff = abs(np.linalg.norm(np.concatenate(numpy_est)) - prev_norm)
        prev_norm = np.linalg.norm(np.concatenate(numpy_est))
        ihvp = [b / scale for b in cur_estimate]
        ihvp = torch.squeeze(torch.stack(ihvp))
        ihvp = [a / num_samples for a in ihvp]
        ihvp = torch.squeeze(torch.stack(ihvp))

    del train_loss, layer_weight, model, hvp, cur_estimate
    torch.cuda.empty_cache()
    return ihvp


def influence(train_data, train_labels, test_data, test_labels, model, layer_weight, n=1, std=0.2,
              criterion=torch.nn.CrossEntropyLoss(), device='cuda:0'):
    eqn_5 = []
    for itr in range(n):
        # print(itr)
        if (n > 1):
            np.random.seed(random.randint(0, 10000000))
            noise = np.random.normal(0, std, test_data.size())
            test_data = test_data.cpu() + noise  # add noise to test data
        if device == 'cuda:0':
            train_data = torch.from_numpy(train_data).float().to(device)
            train_labels = torch.from_numpy(train_labels).long().to(device)
            test_data = torch.from_numpy(test_data).float().to(device)
            test_labels = torch.from_numpy(test_labels).long().to(device)
            model = model.to(device)

        train_mu, train_sig = model(train_data)
        train_loss = model.batch_loss(train_mu, train_sig, torch.nn.functional.one_hot(train_labels, 2))

        test_mu, test_sig = model(test_data)
        test_loss = model.batch_loss(test_mu, test_sig, torch.nn.functional.one_hot(test_labels, 2))

        ihvp = lissa(train_loss, test_loss, layer_weight, model)

        ihvp = ihvp.detach()

        x = train_data
        x.requires_grad = True
        # x_out, x_sig = model(x)
        # x_loss = criterion(x_out, train_labels)
        x_loss = train_loss

        grads = grad(x_loss, layer_weight, create_graph=True)[0]
        grads = grads.squeeze()
        grads = grads.view(1, -1).squeeze()

        infl = (torch.dot(ihvp.view(-1, 1).squeeze(), grads)) / len(train_data)
        i_pert = grad(infl, x, retain_graph=False)
        i_pert = i_pert[0]

        eqn_2 = -infl.detach().cpu().numpy()
        eqn_5.append(np.sum(-i_pert.detach().cpu().numpy(), axis=0))
        model.zero_grad()

        del model, train_data, train_labels, test_labels, layer_weight, ihvp, x, grads, infl, i_pert, eqn_2
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated(0))

    return eqn_5


if __name__ == '__main__':
    main()
