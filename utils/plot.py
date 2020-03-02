import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import operator
import matplotlib.patches as mpatches
import scanpy as sc
import pre
from sklearn import preprocessing

sns.set(style='white', rc={'figure.figsize': (8, 6), 'figure.dpi': 150})
matplotlib.use('TkAgg')
bar = ['c', 'olivedrab', 'darkorange', 'darkseagreen', 'r', 'silver']


def plot_scatter(x_tsne, label_path, out_path, mode, title, xylabel='umap', c_map='Set1'):
    if mode == 'type':
        with open(label_path) as f1:
            f11 = f1.readlines()
        label, color = [], []
        for x in f11:
            label.append(x)
        print(label[0])
        for i in range(len(label)):
            if operator.eq(label[i], '293t\n'):
                color.append(bar[0])
            else:
                color.append(bar[2])
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=color, s=1.5, linewidth=0)
        plt.xlabel(xylabel+'1')
        plt.ylabel(xylabel+'2')
        plt.title(title)

        labels = ['293t', 'jurkat']  # legend标签列表，上面的color即是颜色列表
        color_bar = ['c', 'darkorange']
        patches = [mpatches.Circle((0.1, 0.1), radius=0.51, color=color_bar[i], label="{:s}".format(labels[i])) for i in range(2)]
        # ax.legend(handles=patches)
        plt.savefig(out_path)

    if mode == 'batch':
        with open(label_path)as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                count = row
        color = []
        for i in range(len(count)):
            for j in range(int(count[i])):
                color.append(i+1)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=color, cmap=c_map, s=1.5, linewidth=0)

        labels = ['B_cells', 'CD4', 'CD14+M', 'CD34', 'CD56']  # legend标签列表，上面的color即是颜色列表

        circle = [mpatches.Circle((0.1, 0.1), 0.5, color=bar[i], label="{:s}".format(labels[i])) for i in range(5)]
        plt.title(title)
        # ax.legend(handles=circle)
        # plt.xlabel("dca1")
        # plt.ylabel("dca2")
        # plt.xlabel("tsne1")
        # plt.ylabel("tsne2")
        # plt.xlabel("scvi1")
        # plt.ylabel("scvi2")
        plt.xlabel(xylabel+'1')
        plt.ylabel(xylabel+'2')

        plt.savefig(out_path)


def plot_mark_gene(x, adata, gene_names, thresholds, xy_label, figsize, sub_range, save_path, font_size=14):
    all_gene = adata.var
    index_list, gene_value, color_list, x_list, xy_label_list = [], [], [], [], []
    for name in gene_names:
        index_list.append(pre.find_gene_pos(all_gene, name))
        x_list.append(x)
        xy_label_list.append(xy_label)
    for i in index_list:
        gene_value.append(preprocessing.minmax_scale(pre.cacu_color(adata.X, i).A))
    for k in range(len(gene_value)):
        para_color = []
        for i in range(len(gene_value[k])):
            if gene_value[k][i] < thresholds[k]:
                para_color.append(bar[5])
            else:
                para_color.append(bar[4])
        color_list.append(para_color)
    plot_multi(x_list, color_list, xy_label_list, gene_names, figsize, sub_range, 2, save_path, font_size=font_size)


def plot_multi(x, color, xy_label, title, fig_size, sub_range, size, sava_path, c_map='Set2', font_size=14):
    fig = plt.figure(figsize=fig_size)

    for i in range(len(x)):
        para_sub = sub_range*10+i+1
        ax = fig.add_subplot(para_sub)
        ax.scatter(x[i][:, 0], x[i][:, 1], c=color[i], cmap=c_map, s=size, linewidth=0)
        plt.xlabel(xy_label[i] + '1')
        plt.ylabel(xy_label[i] + '2')
        plt.title(title[i], fontsize=font_size)
    plt.savefig(sava_path)










