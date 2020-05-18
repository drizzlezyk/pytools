import numpy as np
from utils import pre
from network import deep_classifier as dc
from keras.utils import to_categorical

base_path = '/Users/zhongyuanke/Documents/course/data_minig/'
file = base_path + 'result/vae.h5ad'
out_path = base_path + 'result/umap_vae01.png'
test_path = base_path + 'result/test.h5ad'

adata = pre.read_sc_data(file)
label = adata.obs['class']
print(adata.shape)
mid = adata.obsm['X_scxx']
a = set()
y = []
count = -1
for i in range(len(label)):
    if label[i] in a:
        y.append(count)
    else:
        count = count+1
        a.add(label[i])
        y.append(count)
y = np.array(y)
y_train = to_categorical(y)
x_train = mid
test = pre.read_sc_data(test_path)

dc.k_fold_predict(x_train, y, mid.shape[1], 80, 4)
# y = dc.predict(x_train, y_train, 32, 100, test.obsm['X_scxx'])
# result = []
# for i in y:
#     b = i.argmax()
#     result.append(b)
# label_bar = ['MED', 'MGL', 'RHB', 'EPD', 'JPA']
# test_label = []
# for i in result:
#     test_label.append(label_bar[i])
# print(test_label)
