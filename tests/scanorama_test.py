import scanorama
import scanpy as sc


base_path = '/Users/zhongyuanke/data/pbmc/'

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
out_path = '/Users/zhongyuanke/data/result/merge3_scanorama.h5ad'
# files = [file1, file2, file4, file5, file6, file7, file8, file9, file10, file11]
# files = [file1, file2, file4, file5, file6]
files = [file1, file4, file5]
adatas = []
for i in range(len(files)):
    adatas.append(sc.read_10x_mtx(files[i]))

integrated, corrected = scanorama.correct_scanpy(adatas, return_dimred=True)

for i in range(len(integrated)):
    corrected[i].obsm['mid'] = integrated[i]
    print(len(integrated[i]))

adata = corrected[0]
for i in range(1, len(integrated)):
    adata = adata.concatenate(corrected[i])

adata.write_h5ad(out_path)

