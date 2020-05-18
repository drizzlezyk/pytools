from utils import pre as pp
from network.vcca import VCCA
import scanpy as sc

patience = 5
epochs = 8
base_path = '/Users/zhongyuanke/data/'

# file1 = base_path+'cd14_monocytes/hg19'
# file2 = base_path+'cd34/hg19'
# file3 = base_path+'b_cells/hg19'
result_path = '../vcca/'
# file = 'merge_result/merge5.h5ad'
# label_path = 'merge_result/cd14_cd34_b_cells.csv'
label_path = '293t_jurkat_cluster.txt'
batch_path = 'merge_result/merge5.csv'
out_path = 'vcca/293t_vcca.h5ad'
# file = 'merge_result/293t_jurkat.h5ad'
file_x = 'pbmc/293t/hg19'
file_y = 'pbmc/293t_jurkat_50_50/hg19'
adata_x = pp.read_sc_data(base_path+file_x, fmt='10x_mtx')
adata_y = pp.read_sc_data(base_path+file_x, fmt='10x_mtx')
sc.pp.filter_genes(adata_x, min_cells=100)
sc.pp.filter_genes(adata_y, min_cells=100)
net = VCCA(input_size_x=adata_x.shape[1], input_size_y=adata_x.shape[1], path=result_path, private=False)
# net = VAE(input_size=adata_all.shape[1], path=result_path, patience=patience)
net.build()
net.compile()
net.train(adata_x, adata_y, epochs=epochs)

# net.encoder.load_weights(result_path+"vae_weights.h5")
# net.decoder.load_weights(result_path+"decoder_weights.h5")
yadata = net.integrate(adata_x, adata_y, out_path)

# x_tsne = TSNE(n_components=2).fit_transform(yadata.obsm['X_scxx'])

# plot.plot_scatter(x_tsne, base_path+label_path, result_path+'mid07.png', 'batch')