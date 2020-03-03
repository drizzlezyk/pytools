import pre

base_path = '/Users/zhongyuanke/data/pbmc/'
base_path2 = '/Users/zhongyuanke/data/'

# file1 = base_path+'293t/hg19'
# file2 = base_path+'jurkat/hg19'
# file3 = base_path+'293t_jurkat_50_50/hg19'
# out_path = base_path+'merge_result/293t_jurkat.h5ad'
# count_path = base_path+'merge_result/293t_jurkat_batch.csv'
# files = [file1, file2, file3]

# file1 = base_path+'b_cells/hg19'
# file2 = base_path+'cd4_t/hg19'
# file3 = base_path+'cd14_monocytes/hg19'
# file4 = base_path+'cd34/hg19'
# file5 = base_path+'cd56_nk/hg19'
# out_path = base_path+'merge_result/merge5.h5ad'
# count_path = base_path+'merge_result/merge5.csv'
# files = [file1, file2, file3, file4, file5]

# adata = pre.read_sc_data(file1, '10x_mtx')
# adata = pre.filt(adata, 0, 1000)
# adata.write_h5ad(out_path)
# print(adata.shape)

file1 = base_path+'cd19_b/hg19'
file2 = base_path+'cd4_t/hg19'
file3 = base_path + 'cd4_r_t/hg19'
file4 = base_path+'cd14_monocytes/hg19'
file5 = base_path+'cd34/hg19'
file6 = base_path+'cd56_nk/hg19'
file7 = base_path+'cd8_c/hg19'
file8 = base_path+'3k/hg19'
file9 = base_path + '293t/hg19'
file10 = base_path + 'jurkat/hg19'
file11 = base_path + '293t_jurkat_50_50/hg19'

# file10 = base_path + '293t_jurkat_50_50/hg19'
out_path = base_path2 + 'merge_result/merge11.h5ad'
count_path = base_path2 + 'merge_result/merge11.csv'
files = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11]

adata_all, count = pre.merge(files, '10x_mtx')
adata_all.write_h5ad(out_path)
pre.write_count(count, out_path)

