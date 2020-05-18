import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

base_path = '/Users/zhongyuanke/Documents/course/data_minig/'
file = base_path + 'data/pp5i_train.gr.csv'
class_path = base_path + 'data/pp5i_train_class.txt'
result_path = '../data_mining/'
out_path = base_path + 'result/dca.h5ad'

classes = pd.read_csv(class_path, sep=' ', header=None)
classes.drop(index=[0], inplace=True)
classes = np.array(classes).flatten()
batch = []

features = pd.read_csv(file, header=None)
features.drop(index=[0], inplace=True)
features.drop(columns=0, inplace=True)
features = np.array(features).T

print(pearsonr(features, classes))
