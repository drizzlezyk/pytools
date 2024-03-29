from utils import pre as pp
from network.vcca import VCCA
import scanpy as sc
import tensorflow as tf
import numpy as np

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
file = 'scvi/scvi_batch_example_no_filt.h5ad'
file_x = 'pbmc/293t/hg19'
file_y = 'pbmc/293t_jurkat_50_50/hg19'
adata = pp.read_sc_data(base_path+file)
adata_x = pp.read_sc_data(base_path+file_x, fmt='10x_mtx')
adata_y = pp.read_sc_data(base_path+file_y, fmt='10x_mtx')
sc.pp.filter_genes(adata_x, min_cells=500)
sc.pp.filter_genes(adata_y, min_cells=500)
#
x = adata_x.X.toarray()
y = adata_y.X.toarray()
x = x[0:1000, ]
y = y[0:1000, ]
y = np.array(y, dtype='float32')
print(y.shape)
# print(x)
# print(y)
# print(adata.shape)
# sc.pp.filter_genes(adata, min_cells=500)
# sc.pp.filter_cells(adata, min_genes=1000)
# print(adata.shape)

net = VCCA(input_size_x=adata_x.shape[1], inputs_size_y=adata_y.shape[1], path=result_path)
# net.inputs_y = tf.convert_to_tensor(y)
# net = VAE(input_size=adata_all.shape[1], path=result_path, patience=patience)
net.build()
net.compile()


net.train(x, y, epochs=epochs, batch_size=200)

# net.encoder.load_weights(result_path+"vae_weights.h5")
# net.decoder.load_weights(result_path+"decoder_weights.h5")
yadata = net.integrate(adata_x, out_path)
print(yadata)
# x_tsne = TSNE(n_components=2).fit_transform(yadata.obsm['X_scxx'])

# plot.plot_scatter(x_tsne, base_path+label_path, result_path+'mid07.png', 'batch')