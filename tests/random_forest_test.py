from other_algorithm import random_forest as rf
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


base_path = '/Users/zhongyuanke/Documents/course/data_minig/'
file = base_path + 'result/orig.h5ad'
train_dataset, targets = rf.read_adata_h5ad(file)
train_dataset = rf.standard_scale(train_dataset)

# Initiate the Random Forest model from parameters
clf = RandomForestClassifier(random_state=42, n_estimators=200)
clf.fit(train_dataset, targets)

# Use cross validation to get the accuracy of the model
predicts = rf.cross_val(clf, train_dataset, targets)
print(metrics.accuracy_score(targets, predicts))
