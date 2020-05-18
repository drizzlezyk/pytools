import scanpy as sc
from utils import pre, plot

bar = ['c', 'm', 'darkorange', 'darkseagreen', 'r']
c_map = 'Set2'
base_path = '/Users/zhongyuanke/data/'
title = ['Original', 'MSE', 'DCA', 'VAE', 'Scanorama']
xy_label = ['umap', 'mse', 'dac', 'scxx']
fig_size = (18, 4)
f_size = 16

# batch_path = base_path+'merge_result/293t_jurkat_batch.csv'
# label_path = base_path+'293t_jurkat_cluster.txt'
# file_dca = base_path+"result/293t_jurkat_dca_2.h5ad"
# file_mse = base_path+"result/293t_jurkat_mse_2.h5ad"
# file_scxx = base_path+'result/293t_jurkat_scxx_z2.h5ad'
# file_orig = base_path+"merge_result/293t_jurkat.h5ad"
# file_scan = base_path+'result/293t_jurkat_scanorama.h5ad'
# batch_fig_path = base_path+'result/293t_jurkat_batch.png'
# type_fig_path = base_path+'result/293t_jurkat_type.png'
# orig_umap = base_path+'result/293t_jurkat_umap.h5ad'

# file_orig = base_path+'merge_result/merge5.h5ad'
# file_scxx = base_path+'result/merge5_scxx_z2.h5ad'
# file_mse = base_path+'result/merge5_mse_2.h5ad'
# file_dca = base_path+'result/merge5_dca_2.h5ad'
# file_scan = base_path+'result/merge5_scanorama.h5ad'
# batch_path = base_path+'merge_result/merge5.csv'
# batch_fig_path = base_path+'result/test2.png'
# orig_umap = base_path+'result/merge3_umap.h5ad'

# file_orig = base_path+'merge_result/merge9.h5ad'
# file_scan = base_path+'result/merge9_scanorama.h5ad'
# batch_path = base_path+'merge_result/merge9.csv'
# batch_fig_path = base_path+'result/merge9.png'

orig_umap = base_path+'result/merge3_umap.h5ad'
file_scxx = base_path+'result/merge3_scxx_z2.h5ad'
file_mse = base_path+'result/merge3_mse.h5ad'
file_dca = base_path+'result/merge3_dca.h5ad'
file_scan = base_path + 'result/merge3_scanorama.h5ad'
batch_path = base_path+'merge_result/cd14_cd34_b_cells.csv'

batch_fig_path = base_path+'gan/merge3_scanorama.png'
file_gan = base_path+'gan/merge3_gan_20.h5ad'


ctype, type_color, batch_color = [], [], []
# # -------------------------------------------------type
# l = pre.get_label_by_txt(label_path)
# label = []
# for i in l:
#     if operator.eq(i, '293t\n'):
#         type_color.append(bar[0])
#         label.append(0)
#     else:
#         type_color.append(bar[1])
#         label.append(1)
# -------------------------------------------------batch

label = pre.get_label_by_count(batch_path)

adata_orig = sc.read_h5ad(orig_umap)
# sc.pp.filter_genes(adata_orig, min_cells=1500)
adata_dca = sc.read_h5ad(file_dca)
adata_mse = sc.read_h5ad(file_mse)
adata_scxx = sc.read_h5ad(file_scxx)
adata_scan = sc.read_h5ad(file_scan)

x_orig = adata_orig.obsm['umap']
x_mse = adata_mse.obsm['mid']
x_dca = adata_dca.obsm['mid']
x_scvi = adata_scxx.obsm['mid']
x_scan = adata_scan.obsm['umap']
print(x_orig.shape)
print(x_scan.shape)
x_list = [x_orig, x_mse, x_dca, x_scvi]
r = pre.cacu_clustering_metrics(x_list, label, ['sh', 'ch'])
print(r)
color_list = [batch_color, batch_color, batch_color, batch_color]
plot.plot_multi(x_list, color_list, xy_label, title, fig_size, 14, 2, batch_fig_path, 'Set2')




