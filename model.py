import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import GRU
import tensorflow_probability as tfp
from tcn import TCN
import numpy as np
import json
from model_config import params
tfd = tfp.distributions


def negloglik_with_uncert(y_true, y_pred):
    mu = y_pred.loc
    sigma = y_pred.scale
    y_true_mu = y_true[:, 0:1]
    y_true_sigma = y_true[:, 1:params.npb + 1]

    variance = tf.square(sigma) + tf.square(y_true_sigma)
    mse = -0.5 * tf.square(y_true_mu - mu) / variance
    log_term = -0.5 * tf.math.log(2 * np.pi * variance)
    loglikelihood = mse + log_term

    return -loglikelihood


def build_model(model_type, dropout_rate=0.2, learning_rate=0.001, mc_dropout=True):
    input = keras.Input(shape=(params.timesteps, params.n_features))
    masked_input = keras.layers.Masking(mask_value=params.maskval, input_shape=(params.timesteps, params.n_features))(input)

    if model_type == "LSTM":
        hidden = LSTM(50, activation='relu', return_sequences=True)(masked_input)
        hidden = Dropout(dropout_rate)(hidden, training=mc_dropout)
        hidden = BatchNormalization()(hidden)
        hidden = LSTM(50, activation='relu')(hidden)
        hidden = Dropout(dropout_rate)(hidden, training=mc_dropout)
        hidden = BatchNormalization()(hidden)
        opt = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)

    if model_type == "TCN":
        hidden = TCN(100, kernel_size=2, nb_stacks=1, dilations=[1, 2, 4, 8],
                     padding='causal', use_skip_connections=True, dropout_rate=dropout_rate, activation='relu')(
            masked_input, training=mc_dropout)
        opt = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)

    if model_type == "GRU":
        hidden = GRU(50, activation='relu', return_sequences=True)(masked_input)
        hidden = Dropout(dropout_rate)(hidden, training=mc_dropout)
        hidden = BatchNormalization()(hidden)
        hidden = GRU(50, activation='relu')(hidden)
        hidden = Dropout(dropout_rate)(hidden, training=mc_dropout)
        hidden = BatchNormalization()(hidden)
        opt = keras.optimizers.Adam(learning_rate=learning_rate)

    hidden = Dropout(dropout_rate)(hidden, training=mc_dropout)
    hidden = Dense(2)(hidden)
    output = tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])))(hidden)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(optimizer=opt, loss=negloglik_with_uncert)
    return model
