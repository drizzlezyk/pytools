import matplotlib
matplotlib.use('TkAgg')
import scanpy as sc
from utils import pre
from network import autoencoder

base_path='/Users/zhongyuanke/data/'
file1='merge_result/jurkat_293t.h5ad'
write_path='/result/jurkat_293t_dca.csv'
adata = pre.read_sc_data(base_path + file1, 'h5ad')
sc.pp.filter_genes(adata, min_cells=200)
print(adata.shape)
size_factor = pre.cacu_size_factor(adata)
print(size_factor.shape)
autoencoder.train_zinb_model(adata, size_factor)

autoencoder.prediction_zinb_middle(adata, size_factor, base_path + write_path)
