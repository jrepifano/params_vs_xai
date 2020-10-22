import numpy as np
import matplotlib.pyplot as plt

outputs = np.load('evi_outputs_70.npy')
# params = [5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000, 25000, 30000, 35000, 40000]
# params = [5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000]
params = [5, 10, 25, 50, 100, 200]
accs, infl, permute = outputs[:, 0, :], outputs[:, 1, :], outputs[:, 2, :]

accs = np.mean(accs[:70,:],axis=0)
infl = np.mean(infl[:70,:], axis=0)
permute = np.mean(permute[:70,:], axis=0)
plt.plot(params, accs, label='Test Accuracy')
plt.plot(params, infl, label='Influence Accuracy')
plt.plot(params, permute, label='Permutation Importance Accuracy')
plt.legend()
plt.xlabel('Number of Hidden Nodes')
plt.show()

