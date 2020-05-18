import numpy as np
from utils import pre as pp
from network.cvae_network import CVAE

patience = 5
epochs = 8
base_path = '/Users/zhongyuanke/data/'

# file1 = base_path+'cd14_monocytes/hg19'
# file2 = base_path+'cd34/hg19'
# file3 = base_path+'b_cells/hg19'
result_path = '../scxx_result/'
# file = 'merge_result/merge5.h5ad'
# label_path = 'merge_result/cd14_cd34_b_cells.csv'
label_path = '293t_jurkat_cluster.txt'
batch_path = 'merge_result/merge5.csv'
out_path = 'dann_data/scvi_batch_example_scxx_32.h5ad'
# file = 'merge_result/293t_jurkat.h5ad'
file = 'scvi/scvi_batch_example_no_filt.h5ad'

# batch_path = 'merge_result/293t_jurkat_batch.csv'
# out_path = 'result/293t_jurkat_scxx_z2.h5ad'
# files = [file1, file2, file3]
# adata_all, count = pp.merge(files)
adata_all = pp.read_sc_data(base_path+file)
# label = pp.get_label_by_txt(label_path)
# sc.pp.filter_genes(adata_all, min_cells=1)
# with open(base_path+batch_path)as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         count = row
#

b1 = np.array([1, 0])
b2 = np.array([0, 1])
x = [b1, b2]


# b1 = np.array([1, 0, 0])
# b2 = np.array([0, 1, 0])
# b3 = np.array([0, 0, 1])
# x = [b1, b2, b3]

# b1 = np.array([1, 0, 0, 0, 0])
# b2 = np.array([0, 1, 0, 0, 0])
# b3 = np.array([0, 0, 1, 0, 0])
# b4 = np.array([0, 0, 0, 1, 0])
# b5 = np.array([0, 0, 0, 0, 1])
# x = [b1, b2, b3, b4, b5]

batch = []

# for i in range(len(count)):
#     for j in range(int(count[i])):
#         batch.append(x[i])
# batch = np.array(batch)

b = np.array(adata_all.obs['batch'])
b = list(map(int, b))

print(b)
for i in range(adata_all.shape[0]):
    batch.append(x[b[i]])

batch = np.array(batch)

adata_all.obsm['X_batch'] = batch

net = CVAE(input_size=adata_all.shape[1], batches=2, path=result_path, patience=patience)
# net = VAE(input_size=adata_all.shape[1], path=result_path, patience=patience)
net.build()
net.compile()
net.train(adata_all, epochs=epochs)

# net.encoder.load_weights(result_path+"vae_weights.h5")
# net.decoder.load_weights(result_path+"decoder_weights.h5")
yadata = net.integrate(adata_all)

# x_tsne = TSNE(n_components=2).fit_transform(yadata.obsm['X_scxx'])
yadata.write_h5ad(base_path+out_path)
# plot.plot_scatter(x_tsne, base_path+label_path, result_path+'mid07.png', 'batch')




