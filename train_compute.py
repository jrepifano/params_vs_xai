# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:10:59 2020

@author: Jake
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
import smoothInfluence
import time
import os
import gc
from numpy.random import RandomState
from eli5.permutation_importance import get_score_importances

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Vanilla(torch.nn.Module):
    def __init__(self, n_feats, num_nodes):
        super(Vanilla, self).__init__()
        self.linear_1 = torch.nn.Linear(n_feats, num_nodes)
        self.linear_2 = torch.nn.Linear(num_nodes, 2)
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.selu(x)
        x = self.linear_2(x)
        pred = self.softmax(x)
        return pred

    def score(self, X, y):
        X = torch.from_numpy(X).float().to(torch.device('cuda:0'))
        y_pred = self(X)
        pred_lab = torch.argmax(y_pred, dim=1).cpu().numpy()
        total = len(np.where(y == pred_lab)[0])
        accuracy = total / len(X)
        return accuracy


def main():
    print('Total memory allocated: ' + str(torch.cuda.memory_allocated()))
    n_samples = np.random.randint(100, 100000)
    print('Number of Samples in DS: ' + str(n_samples))
    n_feats = np.random.choice([10, 20, 50, 100, 200, 500], 1).item()
    n_clusters = np.random.randint(2, 14)
    sep = 5 * np.random.random_sample()
    hyper = np.random.choice([True, False], 1).item()

    X, y = make_classification(n_samples, n_feats, n_feats // 2, 0, 0, 2, n_clusters, None, 0, sep, True, 0, 1, hyper)
    X, x_test, y, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_test = scaler.transform(x_test)
    device = torch.device('cuda:0')
    if (torch.cuda.is_available()):
        print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

    no_epochs = 100

    accs = []
    infl = []
    permute = []

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    params = [5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000]

    for i in range(len(params)):
        start_time = time.time()
        torch.cuda.empty_cache()
        model = Vanilla(n_feats, params[i])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        if device:
            model.to(device)
            print('Moved to GPU')

        for epoch in range(no_epochs):
            total_train_loss = 0

            model.train()

            image = (X).float().to(device)
            label = y.long().to(device)

            optimizer.zero_grad()

            pred = model(image)

            loss = criterion(pred, label)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            if epoch != 0 and (epoch % 1 == 0):
                print('Epoch: ' + str(epoch) + '/' + str(no_epochs)+', Train Loss: '+str(total_train_loss))
        # validation
        model.eval()
        print(time.time() - start_time)
        image_test = torch.from_numpy(x_test).float().to(device)
        label_test = torch.from_numpy(y_test).long().to(device)

        pred_test = model(image_test)

        test_acc = model.score(x_test, y_test)
        accs.append(test_acc)

        inform_feats = set(range(n_feats // 2))

        eqn_5_smooth = smoothInfluence.influence(image.detach(), label.detach(), image_test.detach(),
                                                 label_test.detach(), model, model.linear_2.weight)
        eqn_5_smooth = np.mean(normalize(np.vstack(eqn_5_smooth)), axis=0)
        loss_acc = len(inform_feats.intersection(set(np.argsort(abs(eqn_5_smooth))[::-1][:n_feats // 2]))) / (
                    n_feats // 2)
        infl.append(loss_acc)

        base_score, score_decreases = get_score_importances(model.score, x_test, y_test)
        perm_importances = np.mean(score_decreases, axis=0)

        perm_acc = len(
            inform_feats.intersection(set(np.argsort(abs(perm_importances))[::-1][:n_feats // 2]))) / (n_feats // 2)
        permute.append(perm_acc)

        print('Inner Loop ' + str(i + 1) + '/' + str(len(params)) + ' Finished')
        del model, image, label, pred_test, pred, loss, optimizer, image_test, label_test
        gc.collect()
        torch.cuda.empty_cache()

        print('Total memory allocated: ' + str(torch.cuda.memory_allocated(device)))

    return np.asarray(test_acc), np.asarray(infl), np.asarray(permute)


if __name__ == "__main__":
    np.random.seed(1234567890)
    torch.manual_seed(1234567890)
    n_experiments = 1000
    params = [5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000]
    outputs = np.empty((n_experiments, 3, len(params)))
    for i in range(n_experiments):
        outputs[i, 0, :], outputs[i, 1, :], outputs[i, 2, :] = main()
        print('Outer Loop ' + str(i + 1) + '/' + str(n_experiments) + ' Finished')
        if i != 0 and ((i < 200 and (i % 10 == 0)) or (i >= 200 and (i % 100 == 0))):
            np.save('outputs_' + str(i) + str('.npy'), outputs)
    # rocs = np.mean(np.squeeze(outputs[:, 0, :]),axis=0)
    # infl = np.mean(np.squeeze(outputs[:, 1, :]), axis=0)
    # permute = np.mean(np.squeeze(outputs[:, 2, :]), axis=0)
    # plt.plot(params, rocs, label='ROC AUC')
    # plt.plot(params, infl, label='Influence Accuracy')
    # plt.plot(params, permute, label='Permutation Importance Accuracy')
    # plt.legend()
    # plt.xlabel('Number of Hidden Nodes')
    # plt.show()
    np.save('outputs_final.npy', outputs)
