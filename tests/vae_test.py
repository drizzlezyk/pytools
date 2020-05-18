import numpy as np
from network.cvae_network import CVAE
import pandas as pd
import anndata

patience = 5
epochs = 20
base_path = '/Users/zhongyuanke/Documents/course/data_minig/'
file = base_path + 'data/pp5i_test.gr.csv'
class_path = base_path + 'data/pp5i_train_class.txt'
result_path = '../data_mining/'
out_path = base_path + 'result/test.h5ad'


classes = pd.read_csv(class_path, sep=' ', header=None)
classes.drop(index=[0], inplace=True)
classes = np.array(classes).flatten()
batch = []

features = pd.read_csv(file, header=None)
features.drop(index=[0], inplace=True)
features.drop(columns=0, inplace=True)
features = np.array(features).T
a = np.array([0])

adata = anndata.AnnData(X=features)

# adata.write_h5ad(out_path)
for i in range(features.shape[0]):
    batch.append(a)
batch = np.array(batch)

adata.obsm['X_batch'] = batch

net = CVAE(input_size=adata.shape[1], batches=1, path=result_path, patience=patience)
# # net = VAE(input_size=adata_all.shape[1], path=result_path, patience=patience)
net.build()
net.compile()
net.train(adata, epochs=epochs)
#yadata.obs['class'] = classes
# # net.encoder.load_weights(result_path+"vae_weights.h5")
# # net.decoder.load_weights(result_path+"decoder_weights.h5")
yadata = net.integrate(adata)

#
# # x_tsne = TSNE(n_components=2).fit_transform(yadata.obsm['X_scxx'])
yadata.write_h5ad(out_path)
# plot.plot_scatter(x_tsne, base_path+label_path, result_path+'mid07.png', 'batch')