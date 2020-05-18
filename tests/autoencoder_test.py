import matplotlib
matplotlib.use('TkAgg')
from utils import pre
from network import autoencoder

base_path = '/Users/zhongyuanke/data/'
# file1 = 'cd14_cd34_b_cells.h5ad'
# file2 = 'merge_result/merge5.h5ad'
# file3 = 'merge_result/293t_jurkat.h5ad'
# write_path = '/result/293t_jurkat_dca_2.h5ad'
file4 = 'merge_result/cd14_cd34_b_cells.h5ad'
write_path = 'result/merge3_dca.h5ad'
label_path = ""
adata = pre.read_sc_data(base_path + file4, 'h5ad')
# sc.pp.filter_genes(adata, min_cells=1000)
epoch = 10
print(adata.shape)
size_factor = pre.cacu_size_factor(adata)
print(size_factor.shape)
autoencoder.train_zinb_model(adata, size_factor, epoch)
adata = autoencoder.prediction_zinb_middle(adata, size_factor, base_path + write_path)
# adata_dca = network.prediction_zinb(adata, size_factor)
adata.write_h5ad(base_path+write_path)





