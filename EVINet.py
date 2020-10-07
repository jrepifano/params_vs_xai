import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from EVI_Layers import EVI_FullyConnected, EVI_Relu, EVI_Softmax
from torch.utils.data import DataLoader, Dataset
####################################################################################################
#
#   Author: Chris Angelini
#
#   Purpose: Extension of Dera et. Al. Bayesian eVI framework into Pytorch
#            The file is used for the creation of the eVI network structure and training loop
#
#   ToDo: Comment
#
####################################################################################################


class data_loader(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        target = self.y[index]
        data_val = self.X[index, :]
        return data_val, target


class EVINet(nn.Module):
    def __init__(self, n_feats, num_nodes):
        super(EVINet, self).__init__()

        self.fullyCon1 = EVI_FullyConnected(n_feats, num_nodes, input_flag=True)
        self.fullyCon2 = EVI_FullyConnected(num_nodes, 2, input_flag=False)
        self.relu = EVI_Relu()
        # self.bn1 = nn.BatchNorm1d(31)
        # self.bn2 = nn.BatchNorm1d(61)
        self.softmax = EVI_Softmax(1)

    def forward(self, x_input):
        flat_x = torch.flatten(x_input, start_dim=1)
        flat_x.requires_grad = True
        mu, sigma = self.fullyCon1.forward(flat_x)
        mu, sigma = self.relu(mu, sigma)
        # mu = self.bn1(mu)
        mu, sigma = self.fullyCon2.forward(mu, sigma)
        mu, sigma = self.softmax.forward(mu, sigma)

        return mu, sigma

    def nll_gaussian(self, y_pred_mean, y_pred_sd, y_test):
        thing = torch.tensor(1e-3)
        y_pred_sd_inv = torch.inverse(y_pred_sd + torch.diag(thing.repeat([self.fullyCon2.out_features])))
        mu_ = y_pred_mean - y_test
        mu_sigma = torch.bmm(mu_.unsqueeze(1), y_pred_sd_inv)
        ms = 0.5 * torch.bmm(mu_sigma, mu_.unsqueeze(2)).squeeze(1) + 0.5 * torch.log(
            torch.det(y_pred_sd + torch.diag(thing.repeat([self.fullyCon2.out_features])))).unsqueeze(1)
        ms = ms.mean()
        return ms

    def batch_loss(self, output_mean, output_sigma, label):
        output_sigma_clamp = torch.clamp(output_sigma, 1e-10, 1e+10)
        tau = 0.002
        log_likelihood = self.nll_gaussian(output_mean, output_sigma_clamp, label)
        loss_value = log_likelihood + tau * (self.fullyCon1.kl_loss_term() + self.fullyCon2.kl_loss_term())
        return loss_value

    def batch_accuracy(self, output_mean, label):
        _, bin = torch.max(output_mean.detach(), dim=1)
        comp = bin == label.detach()
        batch_accuracy = comp.sum().cpu().numpy()/len(label)
        return batch_accuracy

    def score(self, X, y):
        predicted_labels = []
        testset = data_loader(X, y)
        # noinspection PyArgumentList
        testloader = DataLoader(testset, batch_size=256, shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=0, collate_fn=None,
                                pin_memory=False, drop_last=False, timeout=0,
                                worker_init_fn=None)
        for itr, (test_data, test_targets) in enumerate(testloader):
            test_data = test_data.float().to(torch.device('cuda:0'))
            y_pred, sig = self(test_data)
            predicted_batch = torch.argmax(y_pred, dim=1).cpu().numpy()
            predicted_labels.extend(predicted_batch.tolist())

        total = len(np.where(y == predicted_labels)[0])
        accuracy = total / len(X)
        return accuracy
