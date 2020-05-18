import scanpy as sc
from dca.api import dca
from utils import pre

base_path = '/Users/zhongyuanke/data/'
file1 = 'merge_result/jurkat_293t.h5ad'
adata = pre.read_sc_data(base_path + file1)

print('filt')
sc.pp.filter_genes(adata, min_counts=1000)
adata_dca=adata

print('start dca')
dca(adata_dca, threads=1)
print('end dca')
sc.pp.normalize_per_cell(adata_dca)
sc.pp.log1p(adata_dca)
sc.pp.pca(adata_dca)

sc.pp.pca(adata)
sc.pl.pca_scatter(adata, color='Group', size=20, title='Original')
sc.pl.pca_scatter(adata_dca, color='Group', size=20, title='DCA')