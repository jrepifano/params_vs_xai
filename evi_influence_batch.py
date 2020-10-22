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
from torch.utils.data import DataLoader, Dataset


class data_loader(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().to('cuda:0')
        self.y = torch.from_numpy(y).long().to('cuda:0')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        target = self.y[index]
        data_val = self.X[index, :]
        return data_val, target


def hessian_vector_product(ys, xs, v):
    J = grad(ys, xs, create_graph=True)[0]
    grads = grad(J, xs, v, retain_graph=True)
    del J, ys, v
    torch.cuda.empty_cache()
    return grads


def lissa(train_loss, test_loss, layer_weight, model):
    scale = 10
    damping = 0.01
    num_samples = 1
    v = grad(test_loss, layer_weight)[0]
    cur_estimate = v.clone()
    prev_norm = 1
    diff = prev_norm
    count = 0
    while diff > 0.00001 and count < 10000:
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


def influence(train_data, train_labels, test_data, test_labels, model, layer_weight, btchsz=64, n=1, std=0.2,
              criterion=torch.nn.CrossEntropyLoss(), device='cuda:0'):
    eqn_5 = []
    for itr in range(n):
        # print(itr)
        if (n > 1):
            np.random.seed(random.randint(0, 10000000))
            noise = np.random.normal(0, std, test_data.size())
            test_data = test_data.cpu() + noise  # add noise to test data
        # if device == 'cuda:0':
        #     train_data = torch.from_numpy(train_data).float().to(device)
        #     train_labels = torch.from_numpy(train_labels).long().to(device)
        #     test_data = torch.from_numpy(test_data).float().to(device)
        #     test_labels = torch.from_numpy(test_labels).long().to(device)
        #     model = model.to(device)

        trainset = data_loader(train_data, train_labels)
        testset = data_loader(test_data, test_labels)
        # noinspection PyArgumentList
        trainloader = DataLoader(trainset, batch_size=btchsz, shuffle=False)
        testloader = DataLoader(testset, batch_size=btchsz, shuffle=False)

        total_train_loss = 0
        for itr, (train_data_batch, train_labels_batch) in enumerate(trainloader):

            train_mu, train_sig = model(train_data_batch)
            train_loss = model.batch_loss(train_mu, train_sig, torch.nn.functional.one_hot(train_labels_batch, 2).to('cuda:1') )
            total_train_loss += train_loss

        total_test_loss = 0
        for itr, (test_data_batch, test_labels_batch) in enumerate(testloader):
            test_mu, test_sig = model(test_data_batch)
            test_loss = model.batch_loss(test_mu, test_sig, torch.nn.functional.one_hot(test_labels_batch, 2).to('cuda:1'))
            total_test_loss += test_loss


        ihvp = lissa(train_loss, test_loss, layer_weight, model)

        ihvp = ihvp.detach()
        del train_data_batch, train_labels_batch, test_data_batch, test_labels_batch
        torch.cuda.empty_cache()

        for itr, (train_data_batch, train_labels_batch) in enumerate(trainloader):
            train_data_batch.requires_grad = True
            x_out, x_sig = model(train_data_batch)
            x_loss = model.batch_loss(x_out, x_sig, torch.nn.functional.one_hot(train_labels_batch, 2).to('cuda:1'))

            grads = grad(x_loss, layer_weight, create_graph=True)[0]
            grads = grads.squeeze()
            grads = grads.view(1, -1).squeeze()

            infl = (torch.dot(ihvp.view(-1, 1).squeeze(), grads)) / len(train_data)
            i_pert = grad(infl, train_data_batch, retain_graph=False)
            i_pert = i_pert[0]

            eqn_2 = -infl.detach().cpu().numpy()
            eqn_5.append(np.sum(-i_pert.detach().cpu().numpy(), axis=0))
            model.zero_grad()

        del model, train_data, train_labels, test_labels, layer_weight, ihvp, grads, infl, i_pert, eqn_2
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated(0))

    return eqn_5


if __name__ == '__main__':
    main()
