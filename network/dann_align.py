from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import pre
import numpy as np
from keras.utils import to_categorical
import umap
import operator
from network.flip_gradient import flip_gradient


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def plot_embedding(source_x, dann_x, y, d, xy_label='umap', title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    fig = plt.figure(figsize=(12, 10))
    fig.add_subplot(221)
    plt.scatter(source_x[:, 0], source_x[:, 1], c=d, cmap='cool', s=10, alpha=0.6)
    plt.xlabel(xy_label+'1')
    plt.ylabel(xy_label+'2')
    plt.title(title[0])

    fig.add_subplot(222)
    plt.scatter(source_x[:, 0], source_x[:, 1], c=y, cmap='Accent', s=10, alpha=0.6)
    plt.xlabel(xy_label+'1')
    plt.ylabel(xy_label+'2')
    plt.title(title[1])

    fig.add_subplot(223)
    plt.scatter(dann_x[:, 0], dann_x[:, 1], c=d, cmap='cool', s=10, alpha=0.6)
    plt.xlabel(xy_label+'1')
    plt.ylabel(xy_label+'2')
    plt.title(title[2])

    fig.add_subplot(224)
    plt.scatter(dann_x[:, 0], dann_x[:, 1], c=y, cmap='Accent', s=10, alpha=0.6)
    plt.xlabel(xy_label+'1')
    plt.ylabel(xy_label+'2')
    plt.title(title[3])
    plt.show()


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


base_path = '/Users/zhongyuanke/data/'
# source_file = base_path + 'dann_data/merge2.h5ad'
# target_file = base_path + 'pbmc/293t_jurkat_50_50/hg19'
# count_path = base_path + 'dann_data/merge2.csv'
# txt_path = base_path + 'dann_data/293t_jurkat_55_cluster.txt'
#
#
# source_adata = pre.read_sc_data(source_file)
# target_adata = pre.read_sc_data(target_file, '10x_mtx')
# print(source_adata.shape, target_adata.shape)
# source_label = pre.get_label_by_count(count_path)
# print(source_label)
# target_label_txt = pre.get_label_by_txt(txt_path)
# target_label = []
#
# print('start')
# for i in range(len(target_label_txt)):
#     if operator.eq(target_label_txt[i], '293t\n'):
#         target_label.append(0)
#     else:
#         target_label.append(1)
# print('end')

# target_label = target_label[start:end]
# print(start, end)
# f = open(base_path+'dann_data/293t_jurkat_55_cluster.txt', 'w')
#
# for ip in target_label:
#     f.write(ip)
# f.close()
#
# print('end write')
# print(np.array(source_adata.X.A).shape)
# print(np.array(source_label).shape)
merge3 = base_path + 'dann_data/293t_jurkat_scxx_32.h5ad'
label_path = base_path + '293t_jurkat_cluster.txt'

adata = pre.read_sc_data(merge3)
# sc.pp.filter_genes(adata, min_cells=50)
source_data = adata.obsm['mid'][:6143, ]
target_data = adata.obsm['mid'][6143:, ]
label_txt = pre.get_label_by_txt(label_path)
print('start')
label = []
for i in range(len(label_txt)):
    if operator.eq(label_txt[i], '293t\n'):
        label.append(0)
    else:
        label.append(1)
print('end')


source_label = label[:6143]
target_label = label[6143:]
# source_data = source_adata.X.A
np.random.seed(116)
np.random.shuffle(source_data)
np.random.seed(116)
np.random.shuffle(source_label)

source_label = to_categorical(source_label)
target_label = to_categorical(target_label)

num_test = 500
source_train = source_data[0:source_data.shape[0]-num_test, ]
source_train_label = source_label[0:source_data.shape[0]-num_test]
source_test = source_data[:num_test, ]
source_test_label = source_label[:num_test, ]

target_train = target_data[0:target_data.shape[0]-num_test, ]
target_train_label = target_label[0:target_data.shape[0]-num_test]
target_test = target_data[:num_test, ]

target_test_label = target_label[:num_test, ]
print(source_train.shape)
print(source_train_label.shape)

print(target_test.shape)
combined_test_data = np.vstack((source_test, target_test))
print(combined_test_data.shape)
combined_test_labels = np.vstack((source_test_label, target_test_label))
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]), np.tile([0., 1.], [num_test, 1])])


class MNISTModel(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self, input_size, label_size, domain_size, batch_size):
        self._build_model(input_size, label_size, domain_size, batch_size)

    def _build_model(self, input_size, label_size, domain_size, batch_size):
        self.input_size = input_size
        self.X = tf.placeholder(tf.float32, [None, input_size], name='X')
        self.domain_label = tf.placeholder(tf.int32, [None, domain_size], name='domain_label')
        self.class_label = tf.placeholder(tf.float32, [None, label_size])
        self.train = tf.placeholder(tf.bool, [])
        self.l = tf.placeholder(tf.float32, [])

        with tf.variable_scope('feature_extractor'):

            # w_t = weight_variable([input_size, 16])
            # b_t = bias_variable([16])
            # e_t = tf.nn.relu(tf.matmul(self.X, w_t) + b_t, name='encode1')
            # e_t = tf.nn.l2_normalize(e_t)

            w0 = weight_variable([input_size, 16])
            b0 = bias_variable([16])
            e1 = tf.nn.relu(tf.matmul(self.X, w0) + b0, name='encode1')
            e1 = tf.nn.l2_normalize(e1)

            # w1 = weight_variable([16, 8])
            # b1 = bias_variable([8])
            # e2 = tf.nn.relu(tf.matmul(e1, w1) + b1, name='encode2')
            # e2 = tf.nn.l2_normalize(e2)

            w2 = weight_variable([16, 8])
            b2 = bias_variable([8])
            f = tf.nn.relu(tf.matmul(e1, w2) + b2, name='feature')
            f = tf.nn.l2_normalize(f)
            # The domain-invariant feature
            self.feature = f

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size // 3*2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.class_label
            source_labels = lambda: tf.slice(self.class_label, [0, 0], [batch_size // 3*2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            w_fc0 = weight_variable([8, 16])
            b_fc0 = bias_variable([16])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, w_fc0) + b_fc0)

            w_fc1 = weight_variable([16, label_size])
            b_fc1 = bias_variable([label_size])
            logits = tf.matmul(h_fc0, w_fc1) + b_fc1

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)

            d_w_fc0 = weight_variable([8, 16])
            d_b_fc0 = bias_variable([16])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_w_fc0) + d_b_fc0)

            d_w_fc1 = weight_variable([16, domain_size])
            d_b_fc1 = bias_variable([domain_size])
            d_logits = tf.matmul(d_h_fc0, d_w_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain_label)


def train_and_evaluate(training_mode, graph, model, batch_size, epoch=5, verbose=False):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for j in range(epoch):
            # Batch generators
            gen_source_batch = batch_generator(
                [source_train, source_label], batch_size // 3 * 2)
            gen_target_batch = batch_generator(
                [target_train, target_label], batch_size // 3)
            gen_source_only_batch = batch_generator(
                [source_train, source_label], batch_size)
            gen_target_only_batch = batch_generator(
                [target_train, target_label], batch_size)

            domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 3 * 2, 1]),
                                       np.tile([0., 1.], [batch_size // 3, 1])])
            step = source_train.shape[0]//batch_size

            for i in range(1000):
                # Adaptation param and learning rate schedule as described in the paper
                p = float(i) / 1000
                l = 2. / (1. + np.exp(-10. * p)) - 1
                # lr = 0.01 / (1. + 10 * p) ** 0.75
                lr = 0.004
                # print(l, lr)

                # Training step
                if training_mode == 'dann':
                    X0, y0 = next(gen_source_batch)
                    X1, y1 = next(gen_target_batch)
                    X = np.vstack([X0, X1])
                    y = np.vstack([y0, y1])

                    _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                        [dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                        feed_dict={model.X: X, model.class_label: y, model.domain_label: domain_labels,
                                   model.train: True, model.l: l, learning_rate: lr})

                    if verbose and i % 9 == 0:
                        print('loss: {}  d_acc: {}  p_acc: {}  p: {}  l: {}  lr: {}'.format(
                            batch_loss, d_acc, p_acc, p, l, lr))

                elif training_mode == 'source':
                    X, y = next(gen_source_only_batch)
                    _, batch_loss = sess.run([regular_train_op, pred_loss],
                                             feed_dict={model.X: X, model.class_label: y, model.train: False,
                                                        model.l: l, learning_rate: lr})

                elif training_mode == 'target':
                    X, y = next(gen_target_only_batch)
                    _, batch_loss = sess.run([regular_train_op, pred_loss],
                                             feed_dict={model.X: X, model.class_label: y, model.train: False,
                                                    model.l: l, learning_rate: lr})


        # Compute final evaluation on test data
        source_acc = sess.run(label_acc,
                              feed_dict={model.X: source_test, model.class_label: source_test_label,
                                         model.train: False})

        target_acc = sess.run(label_acc,
                              feed_dict={model.X: target_test, model.class_label: target_test_label,
                                         model.train: False})

        test_domain_acc = sess.run(domain_acc,
                                   feed_dict={model.X: combined_test_data,
                                              model.domain_label: combined_test_domain, model.l: 1.0})

        test_emb = sess.run(model.feature, feed_dict={model.X: combined_test_data})

    return source_acc, target_acc, test_domain_acc, test_emb


graph = tf.get_default_graph()
with graph.as_default():
    my_batch_size = 120
    model = MNISTModel(source_train.shape[1], 2, 2, my_batch_size)

    learning_rate = tf.placeholder(tf.float32, [])

    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss

    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain_label, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

    print('\nSource only training')
    source_acc, target_acc, _, source_only_emb = train_and_evaluate('source', graph, model, batch_size=my_batch_size, epoch=5)
    print('Source (MNIST) accuracy:', source_acc)
    print('Target (MNIST-M) accuracy:', target_acc)

    print('\nDomain adaptation training')
    source_acc, target_acc, d_acc, dann_emb = train_and_evaluate('dann', graph, model, batch_size=my_batch_size, epoch=5)
    print('Source (MNIST) accuracy:', source_acc)
    print('Target (MNIST-M) accuracy:', target_acc)
    print('Domain accuracy:', d_acc)
    #
    # tsne = TSNE()
    # source_only_tsne = tsne.fit_transform(source_only_emb)
    #
    # tsne = TSNE()
    # dann_tsne = tsne.fit_transform(dann_emb)
    source_umap = umap.UMAP().fit_transform(source_only_emb)
    dann_umap = umap.UMAP().fit_transform(dann_emb)

title = ['source only batch', 'source only label', 'dann batch', 'dann label']
plot_embedding(source_umap, dann_umap, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'umap',
               title)

