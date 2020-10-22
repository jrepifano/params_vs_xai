# -*- coding: utf-8 -*-
"""
Created on Mon Oct 5

@author: Jake
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
import evi_influence_batch
import time
import os
import gc
from numpy.random import RandomState
import EVINet
from torch.utils.data import DataLoader, Dataset
from eli5.permutation_importance import get_score_importances



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'

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


def main():
    print('Total memory allocated: ' + str(torch.cuda.memory_allocated()))
    n_samples = np.random.randint(100, 100000)
    # n_samples = 100000
    print('Number of Samples in DS: ' + str(n_samples))
    n_feats = np.random.choice([10, 20, 50, 100, 200, 500], 1).item()
    # n_feats = 500
    n_clusters = np.random.randint(2, 14)
    sep = 5 * np.random.random_sample()
    hyper = np.random.choice([True, False], 1).item()

    X, y = make_classification(n_samples, n_feats, n_feats // 2, 0, 0, 2, n_clusters, None, 0, sep, True, 0, 1, hyper)
    X, x_test, y, y_test = train_test_split(X, y, test_size=0.2)

    btchsz = [len(X), len(X), len(X), len(X), len(X), len(X), len(X), len(X), len(X), len(X), 25000, 20000, 10000, 5000]
    params = [5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000, 25000, 30000, 35000, 40000]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_test = scaler.transform(x_test)

    trainset = data_loader(X, y)
    testset = data_loader(x_test, y_test)

    if torch.cuda.is_available():
        print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

    no_epochs = 5

    accs = []
    infl = []
    permute = []

    for i in range(len(params)):
        start_time = time.time()
        torch.cuda.empty_cache()
        iter = i
        model = EVINet.EVINet(n_feats, params[iter],  batch_size=btchsz[iter])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainloader = DataLoader(trainset, batch_size=btchsz[iter], shuffle=False)

        testloader = DataLoader(testset, batch_size=btchsz[iter], shuffle=False)

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(no_epochs):
            total_train_loss = 0
            for batchidx, (train_data, train_targets) in enumerate(trainloader):

                model.train()

                targets_hot = torch.nn.functional.one_hot(train_targets, 2)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=False):
                    pred, sig = model(train_data)

                loss = model.batch_loss(pred, sig, targets_hot.to('cuda:1'))
                total_train_loss += loss.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if epoch != 0 and (epoch % 1 == 0):
                print('Epoch: ' + str(epoch) + '/' + str(no_epochs)+', Train Loss: '+str(total_train_loss))
        print("Total Train Time: "+str(time.time() - start_time))
        # validation
        model.eval()
        test_acc = model.score(x_test, y_test)
        accs.append(test_acc)
        print('Test Accuracy: '+str(test_acc))

        inform_feats = set(range(n_feats // 2))

        model.zero_grad()
        del train_data, train_targets, loss, optimizer
        torch.cuda.empty_cache()
        eqn_5_smooth = evi_influence_batch.influence(X, y, x_test,
                                                     y_test, model, model.fullyCon2.mean_fc.weight, btchsz=btchsz[iter])
        eqn_5_smooth = np.mean(normalize(np.vstack(eqn_5_smooth)), axis=0)
        loss_acc = len(inform_feats.intersection(set(np.argsort(abs(eqn_5_smooth))[::-1][:n_feats // 2]))) / (
                n_feats // 2)
        infl.append(loss_acc)

        start_time = time.time()
        base_score, score_decreases = get_score_importances(model.score, x_test, y_test)
        perm_importances = np.mean(score_decreases, axis=0)
        print("Total Permutation Time: " + str(time.time() - start_time))
        perm_acc = len(
            inform_feats.intersection(set(np.argsort(abs(perm_importances))[::-1][:n_feats // 2]))) / (n_feats // 2)
        permute.append(perm_acc)

        print('Inner Loop ' + str(i + 1) + '/' + str(len(params)) + ' Finished')
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return np.asarray(accs), np.asarray(infl), np.asarray(permute)


if __name__ == "__main__":
    np.random.seed(1234567890)
    torch.manual_seed(1234567890)
    n_experiments = 600
    params = [5, 10, 25, 50, 100, 200]
    outputs = np.empty((n_experiments, 3, len(params)))
    for i in range(n_experiments):
        outputs[i, 0, :], outputs[i, 1, :], outputs[i, 2, :] = main()
        print('Outer Loop ' + str(i + 1) + '/' + str(n_experiments) + ' Finished')
        if i != 0 and ((i < 200 and (i % 10 == 0)) or (i >= 200 and (i % 100 == 0))):
            np.save('evi_outputs_' + str(i) + str('.npy'), outputs)
    np.save('evi_outputs_final.npy', outputs)
