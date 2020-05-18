import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import os
import os.path
import tensorflow as tf
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda, Concatenate, concatenate
from keras.models import Model, Sequential, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers, regularizers, initializers
from keras.utils import plot_model
from keras.losses import mse, binary_crossentropy


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e5)


class VAE:
    def __init__(self, input_size, path="./",
                 y_size=10,
                 method="lognorm",
                 validation_split=0.0,
                 patience=60,
                 deterministic=False):
        self.input_size = input_size
        self.vae = None
        self.inputs = None
        self.outputs = None
        self.path = path
        self.initializers = "glorot_uniform"
        self.method = method
        self.optimizer = optimizers.Adam(lr=0.01)
        self.y_size = y_size
        self.dropout_rate = 0.01
        self.kernel_regularizer = regularizers.l1_l2(l1=0.00, l2=0.00)
        self.validation_split = validation_split
        self.deterministic = deterministic
        callbacks = []
        checkpointer = ModelCheckpoint(filepath=path + "vae_weights.h5", verbose=1, save_best_only=False,
                                       save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='loss', patience=patience)
        tensor_board = TensorBoard(log_dir=path + 'logs/')
        callbacks.append(checkpointer)
        callbacks.append(reduce_lr)
        callbacks.append(early_stop)
        callbacks.append(tensor_board)
        self.callbacks = callbacks

    def build(self):
        # build encoder
        Relu = "relu"
        inputs = Input(shape=(self.input_size,), name='encoder_input')
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  bias_initializer='zeros', name='en_hidden_layer_x1')(inputs)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  name='en_hidden_layer_x2')(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)

        z_mean = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                       name="encoder_mean")(x)
        z_log_var = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                          name="encoder_log_var")(x)
        z = Lambda(sampling, output_shape=(32,), name='hidden_var_z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_mlp')

        latent_inputs = Input(shape=(32,), name='z_sampling')

        # x = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(latent_inputs)
        # x = BatchNormalization(center=True, scale=False)(x)
        # x = Activation(Relu)(x)
        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(z)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        if self.method == "count":
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear")(x)
        elif self.method == "qqnorm":
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear")(x)
        else:
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="softplus")(x)
        decoder = Model(latent_inputs, outputs, name='decoder_mlp')
        if self.deterministic:
            outputs = decoder(encoder(inputs)[0])
        else:
            outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= self.input_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def compile(self):
        self.vae.compile(optimizer=self.optimizer)
        self.vae.summary()

    def train(self, adata, batch_size=256, epochs=300):
        if os.path.isfile(self.path + "vae_weights.h5"):
            self.vae.load_weights(self.path + "vae_weights.h5")
        else:
            self.vae.fit(adata.X, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks,
                         validation_split=self.validation_split, shuffle=True)
            # self.vae.model.save(self.path + "/model_best.h5")

    def integrate(self, xadata, save=True, use_mean=True):
        [z, z_mean, z_log_var, z_batch] = self.encoder.predict(xadata.X)
        if use_mean:
            y_mean = self.decoder.predict(z_mean)
        else:
            y_mean = self.decoder.predict(z_batch)

        xadata.obsm['mid']= z_mean
        return xadata


class CVAE(VAE):
    def __init__(self, batches=2, **kwargs):
        super().__init__(**kwargs)
        self.batches = batches

    def build(self):
        Relu = "relu"
        inputs = Input(shape=(self.input_size,), name='encoder_input')
        inputs_batch = Input(shape=(self.batches,), name='batch_input')
        x = concatenate([inputs, inputs_batch])
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  bias_initializer='zeros', name='en_hidden_layer_x1')(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  name='en_hidden_layer_x2')(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        # x = Dropout(self.dropout_rate)(x)

        # x = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
        #           name='en_hidden_layer_x3')(x)
        # x = BatchNormalization(center=True, scale=False)(x)
        # x = Activation(Relu)(x)
        # x = Dropout(self.dropout_rate)(x)


        z_mean = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                       name="encoder_mean")(x)
        z_log_var = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                          name="encoder_log_var")(x)
        z = Lambda(sampling, output_shape=(32,), name='hidden_var_z')([z_mean, z_log_var])
        encoder = Model([inputs, inputs_batch], [z_mean, z_log_var, z], name='encoder_mlp')

        latent_inputs = Input(shape=(32,), name='z_sampling')
        x = concatenate([latent_inputs, inputs_batch])
        # x = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
        # x = BatchNormalization(center=True, scale=False)(x)
        # x = Activation(Relu)(x)
        # x = Dropout(self.dropout_rate)(x)
        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)

        if self.method == "count":
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear")(x)
        elif self.method == "qqnorm":
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear")(x)
        else:
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="softplus")(x)
        decoder = Model([latent_inputs, inputs_batch], outputs, name='decoder_mlp')

        if self.deterministic:
            outputs = decoder([encoder([inputs, inputs_batch])[0], inputs_batch])
        else:
            outputs = decoder([encoder([inputs, inputs_batch])[2], inputs_batch])
        vae = Model([inputs, inputs_batch], outputs, name='vae_mlp')
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= self.input_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def train(self, adata, batch_size=256, epochs=300):
        if os.path.isfile(self.path + "vae_weights.h5"):
            self.vae.load_weights(self.path + "vae_weights.h5")
        else:
            self.vae.fit([adata.X, adata.obsm['X_batch']], epochs=epochs, batch_size=batch_size,
                         callbacks=self.callbacks, validation_split=self.validation_split, shuffle=True)

    def integrate(self, xadata, save=False, use_mean=False):
        [z_mean, z_log_var, z_sample] = self.encoder.predict([xadata.X, xadata.obsm['X_batch']])
        if use_mean:
            y_mean = self.decoder.predict([z_mean, xadata.obsm['X_batch']])
        else:
            y_mean = self.decoder.predict([z_sample, xadata.obsm['X_batch']])
        # yadata = AnnData(X=y_mean, obs=xadata.obs, var=xadata.var)
        # yadata.raw = xadata.copy()
        xadata.obsm['mid'] = z_mean
        print(z_sample.shape)
        if save:
            xadata.write(self.path + "output.h5ad")
        return xadata

