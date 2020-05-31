from keras.models import Model, load_model
from keras import backend as K
from keras.losses import mse
from keras import optimizers, regularizers, initializers
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda, Concatenate, concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import os.path
import tensorflow as tf


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e5)


class VCCA:
    def __init__(self, input_size_x,inputs_size_y, path="./",
                 y_size=10,
                 method="lognorm",
                 validation_split=0.0,
                 patience=60,
                 deterministic=False):
        self.input_size_x = input_size_x
        self.input_size_y = inputs_size_y
        self.vcca = None
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
        inputs_x = Input(shape=(self.input_size_x,), name='inputs_x')
        inputs_y = Input(shape=(self.input_size_y,), name='inputs_y')

        hx = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  bias_initializer='zeros', name='hx_hidden_layer_x1')(inputs_x)
        hx = BatchNormalization(center=True, scale=False)(hx)
        hx = Activation(Relu)(hx)
        hx = Dropout(self.dropout_rate)(hx)

        hx = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  name='hx_hidden_layer_x2')(hx)
        hx = BatchNormalization(center=True, scale=False)(hx)
        hx = Activation(Relu)(hx)
        hx = Dropout(self.dropout_rate)(hx)

        hx_mean = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                       name="hx_mean")(hx)
        hx_log_var = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                          name="hx_log_var")(hx)
        hx_z = Lambda(sampling, output_shape=(32,), name='hx_z')([hx_mean, hx_log_var])
        encoder_hx = Model(inputs_x, [hx_mean, hx_log_var, hx_z], name='encoder_hx')

        hy = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                   bias_initializer='zeros', name='hy_hidden_layer_x1')(inputs_y)
        hy = BatchNormalization(center=True, scale=False)(hy)
        hy = Activation(Relu)(hy)
        hy = Dropout(self.dropout_rate)(hy)

        hy = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                   name='hy_hidden_layer_x2')(hy)
        hy = BatchNormalization(center=True, scale=False)(hy)
        hy = Activation(Relu)(hy)
        hy = Dropout(self.dropout_rate)(hy)

        hy_mean = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                        name="hy_mean")(hy)
        hy_log_var = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                           name="hy_log_var")(hy)
        hy_z = Lambda(sampling, output_shape=(32,), name='hy_z')([hy_mean, hy_log_var])

        z = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                   bias_initializer='zeros', name='z_hidden_layer_x1')(inputs_x)
        z = BatchNormalization(center=True, scale=False)(z)
        z = Activation(Relu)(z)
        z = Dropout(self.dropout_rate)(z)

        z = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                   name='z_hidden_layer_x2')(z)
        z = BatchNormalization(center=True, scale=False)(z)
        z = Activation(Relu)(z)
        z = Dropout(self.dropout_rate)(z)

        z_mean = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                        name="z_mean")(z)
        z_log_var = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                           name="z_log_var")(z)
        z = Lambda(sampling, output_shape=(32,), name='z')([hx_mean, hx_log_var])
        encoder_z = Model(inputs_x, [z_mean, z_log_var, z], name='encoder_z')


        # x = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(latent_inputs)
        # x = BatchNormalization(center=True, scale=False)(x)
        # x = Activation(Relu)(x)
        latent_x = concatenate([hx_z, z])
        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(latent_x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        outputs_x = Dense(self.input_size_x, kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.initializers, activation="linear")(x)

        latent_y = concatenate([hy_z, z])
        y = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(latent_y)
        y = BatchNormalization(center=True, scale=False)(y)
        y = Activation(Relu)(y)
        y = Dropout(self.dropout_rate)(y)
        y = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(y)
        y = BatchNormalization(center=True, scale=False)(y)
        y = Activation(Relu)(y)
        y = Dropout(self.dropout_rate)(y)
        outputs_y = Dense(self.input_size_y, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear", name='outputs_y')(y)
        # decoder_y = Model(latent_inputs, outputs_y, name='decoder_y')
        # decoder_x = Model(latent_inputs, outputs_x, name='decoder_x')

        # outputs_x = decoder_x(encoder(inputs_x)[0])
        # outputs_y = decoder_y(encoder(inputs_x)[0])

        vcca = Model([inputs_x, inputs_y], [outputs_x, outputs_y], name='vae_mlp')
        reconstruction_loss_x = mse(inputs_x, outputs_x)
        reconstruction_loss_x *= self.input_size_x
        reconstruction_loss_y = mse(inputs_y, outputs_y)
        reconstruction_loss_y *= self.input_size_y
        kl_loss_z = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss_z = -0.5 * K.sum(kl_loss_z, axis=-1)
        kl_loss_hx = 1 + hx_log_var - K.square(hx_mean) - K.exp(hx_log_var)
        kl_loss_hx = -0.5 * K.sum(kl_loss_hx, axis=-1)
        kl_loss_hy = 1 + hy_log_var - K.square(hy_mean) - K.exp(hy_log_var)
        kl_loss_hy = -0.5 * K.sum(kl_loss_hy, axis=-1)

        vae_loss = reconstruction_loss_x + reconstruction_loss_y + kl_loss_z + kl_loss_hx + kl_loss_hy

        vcca.add_loss(vae_loss)
        self.vcca = vcca
        self.encoder_z = encoder_z
        # self.decoder_x = decoder_x

    def compile(self):
        self.vcca.compile(optimizer=self.optimizer)
        self.vcca.summary()

    def train(self, x, y, batch_size=1, epochs=300):
        # if os.path.isfile(self.path + "vae_weights.h5"):
        #     self.vae.load_weights(self.path + "vae_weights.h5")
        # else:
        #
        self.vcca.fit({'inputs_x': x, 'inputs_y': y}, epochs=epochs, batch_size=batch_size,
                         validation_split=self.validation_split, shuffle=True)
            # self.vae.model.save(self.path + "/model_best.h5")

    def integrate(self, xadata, save=True, use_mean=True):
        [z_mean, z_log_var, z] = self.encoder_z.predict(xadata.X)
        xadata.obsm['mid'] = z_mean
        return xadata





