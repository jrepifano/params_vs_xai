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

def hessian_vector_product(ys,xs,v):
    J = grad(ys,xs, create_graph=True)[0]
    grads = grad(J,xs,v,retain_graph=True)
    # J.backward(v,retain_graph=True)
    del J, ys, v
    # return xs.grad
    torch.cuda.empty_cache()
    return grads

def lissa(train_loss,test_loss,layer_weight,model):
    scale = 10
    damping = 0.1
    num_samples = 1
    v = grad(test_loss,layer_weight)[0]
    cur_estimate = v.clone()
    prev_norm = 1
    diff = prev_norm
    count = 0
    while diff > 0.00001:
        hvp = hessian_vector_product(train_loss, layer_weight, cur_estimate)
        cur_estimate = [a+(1-damping)*b-c/scale for (a,b,c) in zip(v,cur_estimate,hvp)]
        cur_estimate = torch.squeeze(torch.stack(cur_estimate))#.view(1,-1)
        model.zero_grad()
        numpy_est = cur_estimate.detach().cpu().numpy()
        numpy_est = numpy_est.reshape(1,-1)
        
        if (count % 100 == 0):
            print("Recursion at depth %s: norm is %.8lf" % (count,np.linalg.norm(np.concatenate(numpy_est))))
        count += 1
        diff = abs(np.linalg.norm(np.concatenate(numpy_est))-prev_norm)
        prev_norm = np.linalg.norm(np.concatenate(numpy_est))
        ihvp = [b/scale for b in cur_estimate]
        ihvp = torch.squeeze(torch.stack(ihvp))
        ihvp = [a/num_samples for a in ihvp]
        ihvp = torch.squeeze(torch.stack(ihvp))
        
    del train_loss, layer_weight, model, hvp, cur_estimate
    torch.cuda.empty_cache()
    return ihvp

def influence(train_data, train_labels, test_data, test_labels, model, layer_weight, n=1, std = 0.2, criterion=torch.nn.CrossEntropyLoss(), device = 'cuda:0'):
    eqn_5 = []
    for itr in range(n):
        # print(itr)
        if(n > 1):
            np.random.seed(random.randint(0,10000000))
            noise = np.random.normal(0,std,test_data.size())
            # print(noise)
            test_data = test_data.cpu()+ noise #add noise to test data
        if device == 'cuda:0':
            train_data = train_data.float().to(device)
            train_labels = train_labels.long().to(device)
            test_data = test_data.float().to(device)
            model = model.to(device)
        # print(test_data)
        train_loss = criterion(model(train_data),train_labels)
        # test_loss = criterion(model(test_data),torch.argmax(model(test_data),axis=1)) # model output
        test_loss = criterion(model(test_data),test_labels)
        # if(len(test_data)==1):
        #     test_loss = criterion(model(test_data.reshape(1,-1).float().to(device)),torch.tensor(1).reshape(1).to(device)) #label 1
        # else:
        #     test_loss = criterion(model(test_data.float().to(device)),torch.ones(len(test_data)).long().to(device)) #label 1
        # print('Test Loss: '+str(test_loss.detach().cpu().numpy()))
        
        # test_loss.backward(create_graph=True)
        
        ihvp = lissa(train_loss,test_loss,layer_weight,model)
        
        ihvp = ihvp.detach()
        # print(torch.cuda.memory_allocated(0))
        x = train_data
        x.requires_grad=True
        x_out = model(x)
        x_loss = criterion(x_out,train_labels)
        # x_loss.backward(create_graph=True)
        # grads = layer_weight.grad
        grads = grad(x_loss,layer_weight,create_graph=True)[0]
        grads = grads.squeeze()
        grads = grads.view(1,-1).squeeze()
        # print(torch.cuda.memory_allocated(0))
        infl = (torch.dot(ihvp.view(-1,1).squeeze(),grads))/len(train_data)
        i_pert = grad(infl,x,retain_graph=False)
        i_pert = i_pert[0]
    
        eqn_2 = -infl.detach().cpu().numpy()
        eqn_5.append(np.sum(-i_pert.detach().cpu().numpy(),axis=0))
        model.zero_grad()
        
        del model, train_data, train_labels, test_labels, layer_weight, ihvp, x, grads, infl, i_pert, eqn_2
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated(0))

    return eqn_5

if __name__ == '__main__':
    main()