import matplotlib
matplotlib.use('TkAgg')
from utils import pre
from network import autoencoder
import pandas as pd
import anndata
import numpy as np

patience = 5
epochs = 20
base_path = '/Users/zhongyuanke/Documents/course/data_minig/'
file = base_path + 'data/pp5i_train.gr.csv'
class_path = base_path + 'data/pp5i_train_class.txt'
result_path = '../data_mining/'
out_path = base_path + 'result/dca.h5ad'

classes = pd.read_csv(class_path, sep=' ', header=None)
classes.drop(index=[0], inplace=True)
classes = np.array(classes).flatten()
batch = []

features = pd.read_csv(file, header=None)
features.drop(index=[0], inplace=True)
features.drop(columns=0, inplace=True)
features = np.array(features).T
a = np.array([0])

adata = anndata.AnnData(X=features)

print(adata.shape)
size_factor = pre.cacu_size_factor(adata)
print(size_factor.shape)
autoencoder.train_zinb_model(adata, size_factor)
adata = autoencoder.prediction_zinb_middle(adata, size_factor, out_path)
adata.obs['class'] = classes
adata.write_h5ad(out_path)
