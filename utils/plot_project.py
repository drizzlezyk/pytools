import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
from sklearn import preprocessing

center = [[-2, -1], [-1, 0.2], [0, -1.5]]
center1 = [[-1.6, -1], [-1.1, 0.8], [-0.2, -1.1]]
center2 = [[-2.2, -0.8], [-0.9, 0.1], [0.3, -1.6]]
std = [0.2, 0.18, 0.25]
std1 = [0.2, 0.15, 0.3]
std2 = [0.25, 0.2, 0.3]
n_samples = 2000
half_s = 1000
font = 20
size = 4

plt.rcParams['figure.figsize'] = (14.0, 4.0)

b1x, b1y = make_blobs(n_samples=half_s, n_features=2, centers=center, cluster_std=std)
b2x, b2y = make_blobs(n_samples=half_s, n_features=2, centers=center, cluster_std=std)
plt.subplot(133)
# plt.title('clusters', fontsize=font)

plt.scatter(b1x[:, 0], b1x[:, 1], c=b1y, cmap='Set1', linewidths=0, s=size)
plt.scatter(b2x[:, 0], b2x[:, 1], c=b2y, cmap='Set1', linewidths=0, s=size)

plt.subplot(132)
# plt.title('aligned', fontsize=font)
plt.scatter(b1x[:, 0], b1x[:, 1], c='mediumaquamarine', linewidths=0, s=size)
plt.scatter(b2x[:, 0], b2x[:, 1], c='gold', linewidths=0, s=size)

plt.subplot(131)
# plt.title('unaligned', fontsize=font)
unali_b1x, unali_b1y = make_blobs(n_samples=half_s, n_features=2, centers=center1, cluster_std=std1)
unali_b2x, unali_b2y = make_blobs(n_samples=half_s, n_features=2, centers=center2, cluster_std=std2)
plt.scatter(unali_b1x[:, 0], unali_b1x[:, 1], c='mediumaquamarine', linewidths=0, s=size)
plt.scatter(unali_b2x[:, 0], unali_b2x[:, 1], c='gold', linewidths=0, s=size)

plt.savefig('/Users/zhongyuanke/Documents/项目申请/fig/batch.png')

