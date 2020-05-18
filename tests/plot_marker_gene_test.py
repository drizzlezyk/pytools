import scanpy as sc
from utils import plot

c_map = 'Purples'
base_path = '/Users/zhongyuanke/data/'
fig_size = (20, 3)
title = ['CD34', 'CD14', 'NKG7', 'CD4', 'CD19']

bar = ['silver', 'r']

file_orig = base_path+'merge_result/merge5.h5ad'
file_scxx = base_path+'result/merge5_scxx_z2.h5ad'
file_mse = base_path+'result/merge5_mse_2.h5ad'
file_dca = base_path+'result/merge5_dca_2.h5ad'
file_scan = base_path+'result/merge5_scanorama.h5ad'
batch_path = base_path+'merge_result/merge5.csv'
fig_path = base_path+'result/merge5_gene_plot_mse.png'
orig_umap = base_path+'result/merge5_umap.h5ad'
adata = sc.read_h5ad(file_orig)
genes = adata.var
adata_orig = sc.read_h5ad(orig_umap)
adata_dca = sc.read_h5ad(file_dca)
adata_mse = sc.read_h5ad(file_mse)
adata_scxx = sc.read_h5ad(file_scxx)
adata_scan = sc.read_h5ad(file_scan)
x_orig = adata_orig.obsm['umap']
x_mse = adata_mse.obsm['mid']
x_dca = adata_dca.obsm['mid']
x_scvi = adata_scxx.obsm['mid']
x_scan = adata_scan.obsm['umap']

gene_names = ['CD34', 'CD14', 'NKG7', 'CD4', 'CD19']
thresholds = [0.05, 0.05, 0.05, 0.01, 0.01]
plot.plot_mark_gene(x_mse, adata, gene_names, thresholds, 'mse', fig_size, 15, fig_path)

