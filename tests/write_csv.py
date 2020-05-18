import csv
import scanpy as sc

base_path = '/Users/zhongyuanke/data/'
batch_path = base_path+'merge_result/293t_jurkat_batch.csv'
file_dca = base_path+"result/293t_jurkat_dca_2.h5ad"
file_mse = base_path+"result/293t_jurkat_mse_2.h5ad"
file_scxx = base_path+'result/293t_jurkat_scxx_z2.h5ad'
orig_umap = base_path+'result/293t_jurkat_umap.h5ad'
file_scan = base_path+'merge_result/293t_jurkat_scanorama.h5ad'

origi_csv = base_path + 'csv/origi.csv'
mse_csv = base_path + 'csv/mse.csv'
dca_csv = base_path + 'csv/dca.csv'
scxx_csv = base_path + 'csv/scxx.csv'
scan_csv = base_path + 'csv/scan.csv'

adata_orig = sc.read_h5ad(orig_umap)
adata_dca = sc.read_h5ad(file_dca)
adata_mse = sc.read_h5ad(file_mse)
adata_scxx = sc.read_h5ad(file_scxx)
adata_scan = sc.read_h5ad(file_scan)

x_orig = adata_orig.obsm['umap']
x_mse = adata_mse.obsm['mid']
x_dca = adata_dca.obsm['mid']
x_scxx = adata_scxx.obsm['mid']
x_scan = adata_scan.obsm['umap']

with open(origi_csv, "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(x_orig[:, 0])
    csv_writer.writerow(x_orig[:, 1])

with open(mse_csv, "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(x_mse[:, 0])
    csv_writer.writerow(x_mse[:, 1])

with open(dca_csv, "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(x_dca[:, 0])
    csv_writer.writerow(x_dca[:, 1])

with open(scxx_csv, "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(x_scxx[:, 0])
    csv_writer.writerow(x_scxx[:, 1])

with open(scan_csv, "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(x_scan[:, 0])
    csv_writer.writerow(x_scan[:, 1])
