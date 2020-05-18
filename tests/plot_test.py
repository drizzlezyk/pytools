import scanpy as sc
from utils import plot
import umap

base_path = '/Users/zhongyuanke/data/'
# label_path = 'merge_result/cd14_cd34_b_cells.csv'
# file_dca = "result/cd14_cd34_b_cells_dca.csv"
# file_mse = 'result/cd14_cd34_b_cells_mse.csv'
# file_orig = "cd14_cd34_b_cells.h5ad"
# out_path = 'result/cd14_cd34_b_mse03.png'
#
batch_path = base_path+'merge_result/293t_jurkat_batch.csv'
label_path = base_path+'293t_jurkat_cluster.txt'
file_dca = base_path+"result/293t_jurkat_dca_2.h5ad"
file_mse = base_path+"result/293t_jurkat_mse_2.h5ad"
file_scxx = base_path+'result/293t_jurkat_scxx_z2.h5ad'
file_orig = base_path+"merge_result/293t_jurkat.h5ad"
file_gan = base_path + 'gan/293t_jurkat_gan_200.h5ad'
scan_path = base_path+'filt/293t_jurkat_scanorama.h5ad'
out_path = base_path+'gan/293t_jurkat_gan_200.png'

# file_orig = base_path+'merge_result/cd14_cd34_b_cells.h5ad'
# file_scxx = base_path+'result/merge5_scxx_z2.h5ad'
# file_mse = base_path+'result/merge5_mse.h5ad'
# file_dca = base_path+'result/merge5_dca.h5ad'
# file_scan = base_path + 'result/merge9_scanorama.h5ad'
# label_path = base_path+'merge_result/cd14_cd34_b_cells.csv'
# out_path = base_path+'gan/merge3_gan_noc.png'
# file_cgan = base_path+'gan/merge3_gan_10.h5ad'
# file_gan = base_path + 'gan/merge3_no_c.h5ad'
# file_umap = base_path + 'result/merge3_umap.h5ad'

# file_orig = base_path+'merge_result/cd14_cd34_b_cells.h5ad'
# file_scxx = base_path+'result/merge3_scxx_z2.h5ad'
# file_mse = base_path+'result/merge3_mse.h5ad'
# file_dca = base_path+'result/merge3_dca.h5ad'
# file_scan = base_path + 'result/merge3_scanorama.h5ad'
# label_path = base_path+'merge_result/cd14_cd34_b_cells.csv'
#
# out_path = base_path+'gan/merge3_scanorama.png'
# file_gan = base_path+'gan/merge3_gan_20.h5ad'
# file_umap = base_path + 'result/merge3_umap.h5ad'
# file_scvi = base_path+'scvi/scvi_example.h5ad'

adata = sc.read_h5ad(file_gan)
# print(adata.shape)
# a_dca = anndata.AnnData(adata.obsm['mid'])
# a_dca.var = adata.var
#
#
# plot.plot_mark_gene(a_dca, 'CD8A')
print('start ---')
#
# with open(label_path)as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         count = row
# label = []
# for i in range(len(count)):
#     for j in range(int(count[i])):
#         label.append(i)
# sc.pp.filter_genes(adata, min_cells=2500)
# print(adata.shape)
# x = adata.obsm['mid']
# sc.pp.filter_genes(adata, min_cells=100)
x = adata.obsm['mid']
# x_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(adata.X)n_neighbors=5, min_dist=0.5, n_components=2
x_tsne = umap.UMAP().fit_transform(x)
# x_tsne = adata.obsm['mid']
# adata.obsm['umap'] = x_tsne
#
# adata.write_h5ad(file_umap)
print('end')
plot.plot_scatter(x_tsne, label_path, out_path, 'type', 'gan', 'umap', 'tab20c')
# label = pre.get_label_by_count(label_path, 'count')
# sh_scxx = metrics.silhouette_score(x_tsne, label, metric='euclidean')
# print(sh_scxx)



