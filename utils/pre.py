import numpy as np
import matplotlib
import csv
import scanpy as sc
from scipy.sparse import csr_matrix
matplotlib.use('TkAgg')


def read_sc_data(input_file, fmt='h5ad', backed=None, transpose=False, sparse=False, delimiter=" "):
    if fmt == '10x_h5':
        adata = sc.read_10x_h5(input_file)
    elif fmt == '10x_mtx':
        adata = sc.read_10x_mtx(input_file)
    elif fmt == "mtx":
        adata = sc.read_mtx(input_file)
    elif fmt == 'h5ad':
        adata = sc.read_h5ad(input_file, backed=backed)
    elif fmt == "csv":
        adata = sc.read_csv(input_file)
    elif fmt == "txt":
        adata = sc.read_text(input_file, delimiter=delimiter)
    elif fmt == "tsv":
        adata = sc.read_text(input_file, delimiter="\t")
    else:
        raise ValueError('`format` needs to be \'10x_h5\' or \'10x_mtx\'')
    if transpose:
        adata = adata.transpose()
    if sparse:
        adata.X = csr_matrix(adata.X,dtype='float32')
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


def cacu_size_factor(adata):
    return adata.X.sum(axis=1)


def get_label_by_count(label_path):
    label = []
    with open(label_path)as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            count = row
    for i in range(len(count)):
        for j in range(int(count[i])):
            label.append(i)
    label = np.array(label)
    return label


def merge(files, type='h5ad'):
    adata_all = read_sc_data(files[0], type)
    sample_count = []
    sample_count.append(adata_all.shape[0])
    for i in range(len(files)-1):
        adata = read_sc_data(files[i+1], type)
        sample_count.append(adata.shape[0])
        adata_all = adata_all.concatenate(adata)
    return adata_all, sample_count


def filt(adata, min_c, min_g):
    sc.pp.filter_genes(adata, min_cells=min_c)
    sc.pp.filter_cells(adata, min_genes=min_g)
    return adata


def write_merge_label(count,labels,path):
    res=[]
    for i in range(len(count)):
        for j in range(count[i]):
          res.append(labels[i])
    print(res[1])
    with open(path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(res)
    f.close()


def find_gene_pos(genes, gene):
    genes['i'] = range(genes.shape[0])
    return genes.loc[gene]['i']


def cacu_color(X, i):
    return X[:, i]


def get_label_by_txt(txtpath):
    label = []
    with open(txtpath) as label_f:
        f11 = label_f.readlines()
    for x in f11:
        label.append(x)
    return label
