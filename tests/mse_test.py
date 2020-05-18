import matplotlib
matplotlib.use('TkAgg')
import scanpy as sc
from utils import pre
import MSE_AE

base_path='/Users/zhongyuanke/data/'

# file1='cd14_cd34_b_cells.h5ad'
# write_path='/result/cd14_cd34_b_cells_mse.csv'

file_path = base_path + "merge_result/merge5.h5ad"
file1 = 'merge_result/293t_jurkat.h5ad'
out_path = '/result/merge5_mse_2.h5ad'
# out_path = 'result/merge5_mse.h5ad'

adata = pre.read_sc_data(file_path, 'h5ad')
sc.pp.filter_genes(adata, min_cells=500)

MSE_AE.train_mse_model(adata, 20)
adata = MSE_AE.prediction_mse_middle(adata, base_path+out_path)
adata.write_h5ad(base_path+out_path)