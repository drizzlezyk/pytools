from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda
from keras.models import Model, load_model, Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.utils import to_categorical

color_bar = ['c', 'olivedrab', 'darkorange']


def predict(x_train, y_train, input_size, epoch_num, test):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    px_train, x_test, py_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    """ 训练网络 """
    history = model.fit(px_train, py_train, epochs=epoch_num, batch_size=6, validation_data=(x_test, y_test))
    y = model.predict(test)

    """ 展示结果 """
    plt.rcParams['figure.figsize'] = (12.0, 5.0)
    plt.subplot(121)
    plt.plot(history.history['acc'], c=color_bar[0])
    plt.plot(history.history['val_acc'], c=color_bar[2])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'], c=color_bar[0])
    plt.plot(history.history['val_loss'], c=color_bar[2])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return y

def k_fold_predict(x_train, y, input_size, epoch_num, k):
    seed = 7
    np.random.seed(seed)
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(x_train, y):
        # create model
        y_train = to_categorical(y)
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(input_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(x_train[train], y_train[train], epochs=epoch_num, batch_size=10, verbose=0)
        # evaluate the model
        scores = model.evaluate(x_train[test], y_train[test], verbose=0)
        print("%s: %.4f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.4f%% (+/- %.4f%%)" % (np.mean(cvscores), np.std(cvscores)))
