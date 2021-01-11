import time
import torch
import numpy as np
import random
from torch.autograd import grad


def hessian_vector_product(ys, xs, v):
    J = grad(ys,xs, create_graph=True)[0]
    grads = grad(J,xs,v,retain_graph=True)
    del J, ys, v
    torch.cuda.empty_cache()
    return grads


def lissa(train_pred_1, test_pred_1, layer_weight, model):
    scale = 10
    damping = 0.01
    num_samples = 1
    v = grad(test_pred_1, layer_weight)[0]
    cur_estimate = v.clone()
    prev_norm = 1
    diff = prev_norm
    count = 0
    while diff > 0.00001:
        hvp = hessian_vector_product(train_pred_1, layer_weight, cur_estimate)
        cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, cur_estimate, hvp)]
        cur_estimate = torch.squeeze(torch.stack(cur_estimate))  # .view(1,-1)
        model.zero_grad()
        numpy_est = cur_estimate.detach().cpu().numpy()
        numpy_est = numpy_est.reshape(1, -1)

        # if (count % 100 == 0):
        #     print("Recursion at depth %s: norm is %.8lf" % (count, np.linalg.norm(np.concatenate(numpy_est))))
        count += 1
        diff = abs(np.linalg.norm(np.concatenate(numpy_est)) - prev_norm)
        prev_norm = np.linalg.norm(np.concatenate(numpy_est))
        ihvp = [b / scale for b in cur_estimate]
        ihvp = torch.squeeze(torch.stack(ihvp))
        ihvp = [a / num_samples for a in ihvp]
        ihvp = torch.squeeze(torch.stack(ihvp))

    return ihvp


def influence(train_data, train_labels, test_data, model, layer_weight, n=10, std=0.2,
              criterion=torch.nn.CrossEntropyLoss(), device='cuda'):
    eqn_5 = []
    for itr in range(n):
        if (n > 1):
            np.random.seed(random.randint(0, 10000000))
            noise = np.random.normal(0, std, test_data.size())
            test_data = test_data.cpu() + noise  # add noise to test data
        if device == 'cuda':
            train_data.to(device)
            train_labels.to(device)
            test_data.to(device)
            model.to(device)

        train_pred_1 = torch.mean(model(train_data),axis=0)[1]

        if (len(test_data) == 1):
            test_pred_1 = model(test_data.reshape(1, -1).float().to(device))[0][1]
        else:
            test_pred_1 = torch.mean(model(test_data.float().to(device)), axis=0)[1]

        ihvp = lissa(train_pred_1, test_pred_1, layer_weight, model)

        ihvp = ihvp.detach()

        x = train_data
        x.requires_grad = True
        x_pred = torch.mean(model(x),axis=0)[1]
        grads = grad(x_pred, layer_weight, create_graph=True)[0]
        grads = grads.squeeze()
        grads = grads.view(1, -1).squeeze()

        infl = (torch.dot(ihvp.view(-1, 1).squeeze(), grads)) / len(train_data)
        i_pert = grad(infl, x, retain_graph=True)
        i_pert = i_pert[0]

        # eqn_2 = -infl.detach().cpu().numpy()
        eqn_5.append(np.sum(-i_pert.detach().cpu().numpy(), axis=0))

        model.zero_grad()

    del model, train_data, train_labels, layer_weight, ihvp, x, grads, infl, i_pert
    torch.cuda.empty_cache()

    return eqn_5