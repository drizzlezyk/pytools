import pre
import plot

colormap = 'coolwarm'
base_path = '/Users/zhongyuanke/Documents/course/data_minig/'
file = base_path + 'result/orig.h5ad'
out_path = base_path + 'result/expression.png'

gene_index = [3211, 5609, 3391, 6093, 6658, 4009, 492, 4444, 2345, 2453, 5362, 2709, 5204, 726, 489, 5108, 5420, 4312,
               2189, 2436, 2386, 642, 5306, 3080, 4531, 4154, 2262, 6336, 2127, 256, 1720, 4683, 5136, 1995, 815, 6411,
               807, 1700, 5102, 2751, 2658, 4507, 236, 2039, 2532, 3602, 862, 2947, 5132, 2546]

adata = pre.read_sc_data(file)
label = adata.obs['class']
title = 'z_score normalize'
plot.plot_expression_matrix(adata.X, gene_index, title, out_path, 'zscore', cmap=colormap)


