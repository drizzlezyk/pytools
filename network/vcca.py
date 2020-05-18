import keras
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda
from keras.regularizers import l1_l2
from keras.models import Model, load_model
from keras import backend
from keras.losses import mse
from keras import optimizers, regularizers, initializers
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda, Concatenate, concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import os.path
import tensorflow as tf


def sampling(args):
    z_mean, z_log_var = args
    batch = backend.shape(z_mean)[0]
    dim = backend.int_shape(z_mean)[1]
    epsilon = backend.random_normal(shape=(batch, dim))
    return z_mean + backend.exp(0.5 * z_log_var) * epsilon


MeanAct = lambda x: tf.clip_by_value(backend.exp(x), 1e-5, 1e5)


class VCCA():
    def __init__(self, input_size_x, input_size_y, validation_split=0, private=True, path='./vcca_result/', patience=60,):
        self.encoder_hx = None
        self.encoder_hy = None
        self.encoder_z = None
        self.decoder_x = None
        self.decoder_y = None
        self.vcca =None
        self.path = '../vcca_result/'
        self.optimizer = optimizers.Adam(lr=0.01)
        self.kernel_regularizer = regularizers.l1_l2(l1=0.00, l2=0.00)
        self.validation_split = validation_split
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        self.private = private
        self.inputs_y = 0
        self.initializers = "glorot_uniform"

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
        Relu = 'relu'
        KL_hx_loss = 0
        KL_hy_loss = 0
        hx_z = tf.placeholder(tf.float32)
        hy_z = tf.placeholder(tf.float32)
        inputs_x = Input(shape=(self.input_size_x,), name="x")

        if self.private:
            self.inputs_y = Input(shape=(self.input_size_y,), name="y")
            hx = Dense(128, kernel_regularizer=l1_l2(l1=0., l2=0.), name="encoder_hx1")(inputs_x)
            hx = BatchNormalization(center=True, scale=False)(hx)
            hx = Activation(Relu, name="activation_hx_1")(hx)
            hx = Dense(64, kernel_regularizer=l1_l2(l1=0., l2=0.), name="encoder_hx2")(hx)
            hx = BatchNormalization(center=True, scale=False)(hx)
            hx = Activation(Relu, name="activation_hx_2")(hx)
            hx_mean = Dense(32, kernel_regularizer=l1_l2(l1=0., l2=0.), name="mean_hx")(hx)
            hx_var = Dense(32, kernel_regularizer=l1_l2(l1=0., l2=0.), name="var_hx")(hx)
            hx_z = Lambda(sampling, output_shape=(32,), name='sample_hx_z')([hx_mean, hx_var])
            self.encoder_hx = Model(inputs=[inputs_x], outputs=hx_z)

            hy = Dense(128, kernel_regularizer=l1_l2(l1=0., l2=0.), name="encoder_hy1")(self.inputs_y)
            hy = BatchNormalization(center=True, scale=False)(hy)
            hy = Activation(Relu, name="activation_hy_1")(hy)
            hy = Dense(64, kernel_regularizer=l1_l2(l1=0., l2=0.), name="encoder_hy2")(hy)
            hy = BatchNormalization(center=True, scale=False)(hy)
            hy = Activation(Relu, name="activation_hy_2")(hy)
            hy_mean = Dense(32, kernel_regularizer=l1_l2(l1=0., l2=0.), name="center_hy_mean")(hy)
            hy_var = Dense(32, kernel_regularizer=l1_l2(l1=0., l2=0.), name="center_hy_var")(hy)
            hy_z = Lambda(sampling, output_shape=(32,), name='sample_hy_z')([hy_mean, hy_var])
            self.encoder_hy = Model(inputs=[self.inputs_y], outputs=hy_z)

            KL_hx_loss = -0.5 * backend.sum(1 + hx_var - backend.square(hx_mean) - backend.exp(hx_var), axis=-1)
            KL_hy_loss = -0.5 * backend.sum(1 + hy_var - backend.square(hy_mean) - backend.exp(hy_var), axis=-1)

        z1 = Dense(128, kernel_regularizer=l1_l2(l1=0., l2=0.), name="encoder_z_mean_1")(inputs_x)
        z1 = BatchNormalization(center=True, scale=False)(z1)
        z1 = Activation(Relu, name="activation_z_mean_1")(z1)
        z1 = Dense(64, kernel_regularizer=l1_l2(l1=0., l2=0.), name="encoder_z_mean_2")(z1)
        z1 = BatchNormalization(center=True, scale=False)(z1)
        z1 = Activation(Relu, name="activation_z_mean_2")(z1)
        z_mean = Dense(32, kernel_regularizer=l1_l2(l1=0., l2=0.), name="center_z_mean")(z1)

        z2 = Dense(128, kernel_regularizer=l1_l2(l1=0., l2=0.), name="encoder_z_var_1")(inputs_x)
        z2 = BatchNormalization(center=True, scale=False)(z2)
        z2 = Activation(Relu, name="activation_z_var_1")(z2)
        z2 = Dense(64, kernel_regularizer=l1_l2(l1=0., l2=0.), name="encoder_z_var_2")(z2)
        z2 = BatchNormalization(center=True, scale=False)(z2)
        z2 = Activation(Relu, name="activation_z_var_2")(z2)
        z_var = Dense(32, kernel_regularizer=l1_l2(l1=0., l2=0.), name="center_z_var")(z2)
        z = Lambda(sampling, output_shape=(32,), name='hidden_var_z')([z_mean, z_var])
        print(z)
        self.encoder_z = Model(inputs=inputs_x, outputs=z_mean)

        latent_inputs_z = Input(shape=(32,), name='latent_input_z')
        if self.private:

            latent_inputs_x = Input(shape=(32,), name='latent_inputs_x')
            latent_inputs_y = Input(shape=(32,), name='latent_inputs_y')
            input_decoder_x = concatenate([latent_inputs_x, latent_inputs_z])
            input_decoder_y = concatenate([latent_inputs_y, latent_inputs_z])
            x = Dense(64, kernel_regularizer=l1_l2(l1=0., l2=0.), kernel_initializer=self.initializers, name="decoder_x_1")(input_decoder_x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu, name="activation_x_1")(x)
            x = Dense(128, kernel_regularizer=l1_l2(l1=0., l2=0.), name="decoder_x_2")(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu, name="activation_x_2")(x)
            output_x = Dense(self.input_size_x, kernel_regularizer=l1_l2(l1=0., l2=0.), name="decoder_x_3")(x)
            self.decoder_x = Model(inputs=[latent_inputs_x, latent_inputs_z], outputs=output_x)

            y = Dense(64, kernel_regularizer=l1_l2(l1=0., l2=0.), name="decoder_y_1")(input_decoder_y)
            y = BatchNormalization(center=True, scale=False)(y)
            y = Activation(Relu, name="activation_y_1")(y)
            y = Dense(128, kernel_regularizer=l1_l2(l1=0., l2=0.), name="decoder_y_2")(y)
            y = BatchNormalization(center=True, scale=False)(y)
            y = Activation(Relu, name="activation_y_2")(y)
            output_y = Dense(self.input_size_y, kernel_regularizer=l1_l2(l1=0., l2=0.), name="decoder_y_3")(y)
            self.decoder_y = Model(inputs=[latent_inputs_y, latent_inputs_z], outputs=output_y)

        else:
            x = Dense(64, kernel_regularizer=l1_l2(l1=0., l2=0.), kernel_initializer=self.initializers, name="decoder_x_1")(latent_inputs_z)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu, name="activation_x_1")(x)
            x = Dense(128, kernel_regularizer=l1_l2(l1=0., l2=0.), name="decoder_x_2")(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu, name="activation_x_2")(x)
            output_x = Dense(self.input_size_x, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear")(x)
            self.decoder_x = Model(latent_inputs_z, output_x)

            y = Dense(64, kernel_regularizer=l1_l2(l1=0., l2=0.), name="decoder_y_1")(latent_inputs_z)
            y = BatchNormalization(center=True, scale=False)(y)
            y = Activation(Relu, name="activation_y_1")(y)
            y = Dense(128, kernel_regularizer=l1_l2(l1=0., l2=0.), name="decoder_y_2")(y)
            y = BatchNormalization(center=True, scale=False)(y)
            y = Activation(Relu, name="activation_y_2")(y)
            output_y = Dense(self.input_size_y, kernel_regularizer=l1_l2(l1=0., l2=0.), name="decoder_y_3")(y)
            self.decoder_y = Model(inputs=latent_inputs_z, outputs=output_y)

        reconstruction_loss1 = mse(inputs_x, output_x)
        reconstruction_loss2 = mse(self.inputs_y, output_y)

        KLz_loss = -0.5 * backend.sum(1 + z_var - backend.square(z_mean) - backend.exp(z_var), axis=-1)
        if self.private:
            loss = KL_hx_loss+KL_hy_loss+KLz_loss+reconstruction_loss1+reconstruction_loss2
        else:
            loss = KLz_loss+reconstruction_loss1+reconstruction_loss2
        if self.private:

            output1 = self.decoder_x([self.encoder_hx(inputs_x), self.encoder_z(inputs_x)])
            output2 = self.decoder_y([self.encoder_hy(self.inputs_y), self.encoder_z(inputs_x)])
            vcca = Model([inputs_x, self.inputs_y], [output1, output2])
            vcca.add_loss(loss)
            self.vcca = vcca
        else:
            output1 = self.decoder_x(self.encoder_z(inputs_x))
            output2 = self.decoder_y(self.encoder_z(inputs_x))
            vcca = Model(inputs_x, [output1, output2])
            vcca.add_loss(loss)
            self.vcca = vcca

    def compile(self):
        self.vcca.compile(optimizer=self.optimizer)
        self.vcca.summary()

    def train(self, x, y, batch_size=256, epochs=300):
        if os.path.isfile(self.path + "vcca_weights.h5"):
            self.vcca.load_weights(self.path + "vcca_weights.h5")
        else:
            if self.private:
                self.vcca.fit([x, y], epochs=epochs, batch_size=batch_size, callbacks=self.callbacks,
                              validation_split=self.validation_split, shuffle=True)
            else:

                self.vcca.fit(x, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks,
                              validation_split=self.validation_split, shuffle=True)

    def integrate(self, xadata, yadata, out_path, save=False):
        [z_sample] = self.encoder_z.predict(xadata.X)
        xadata.obsm['mid'] = z_sample
        print(z_sample.shape)
        if save:
            xadata.write(out_path)
        return xadata



