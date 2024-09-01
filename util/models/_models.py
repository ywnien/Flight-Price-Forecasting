import time
from functools import reduce

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import (LSTM, Activation, Concatenate, Conv1D, Dense,
                          Embedding, Input, LayerNormalization, Permute,
                          RepeatVector, Reshape, TimeDistributed, add,
                          concatenate, Softmax, Dropout, Flatten)
from keras.metrics import mean_absolute_error
from keras.models import Model
import tensorflow as tf

from ..snapshot import LAGS, N_STEPS

NUM_CLASSES = 11
embed_dim = 2
input_names = ["encoded", "fare_basis"]
output_names = ["fare", "seat"]


def create_inputs():
    encoded = Input(shape=(LAGS, 34), name=input_names[0])
    fare_basis = Input(shape=(LAGS, 1), name=input_names[1])
    # encoded = Input(shape=(LAGS, 34), batch_size=32, name=input_names[0])
    # fare_basis = Input(shape=(LAGS, 1), batch_size=32, name=input_names[1])

    embedded = Embedding(input_dim=300, output_dim=embed_dim)(fare_basis)
    embedded = Reshape((LAGS, embed_dim))(embedded)
    x = Concatenate(axis=-1)([encoded, embedded])
    return [encoded, fare_basis], x

# model_comment = "LSTM_encoder_decoder"
def LSTM_ed(**kwargs):
    inputs, x = create_inputs()
    units = x.shape[-1]

    rnn_configs = {"units": units, "kernel_regularizer": "l2"} | kwargs

    _lstm_output = LayerNormalization()(
        LSTM(**rnn_configs, dropout=0.3, return_sequences=True)(x)
    )
    _lstm_output = LayerNormalization()(
        add([LSTM(**rnn_configs, dropout=0.3, return_sequences=True)(_lstm_output), x])
    )
    _lstm_output = LayerNormalization()(
        LSTM(**rnn_configs)(_lstm_output)
    )
    lstm_encode = RepeatVector(N_STEPS)(_lstm_output)

    _lstm_decode = LayerNormalization()(
        LSTM(**rnn_configs, return_sequences=True)(lstm_encode)
    )
    lstm_output = LayerNormalization()(
        add([LSTM(**rnn_configs, return_sequences=True)(_lstm_decode), lstm_encode])
    )

    # Regression output
    fare = TimeDistributed(
        Dense(1, activation='linear'), name=output_names[0]
    )(lstm_output)
    # Classification output
    seat = TimeDistributed(
        Dense(11, activation='softmax'), name=output_names[0]
    )(lstm_output)

    model = Model(inputs=inputs, outputs=[fare, seat])

    model.compile(
        optimizer='adam',
        loss={'fare': 'mse', 'seat': 'sparse_categorical_crossentropy'},
        metrics={'fare': 'mae'}
    )
    model.summary()
    return model


def _causal_output(x, lags=LAGS):
    fare = TimeDistributed(Dense(N_STEPS), name="fare")(x)

    clf_classes = concatenate(
        [
            TimeDistributed(Dense(N_STEPS), name=f"class_{i}")(x)
            for i in range(NUM_CLASSES)
        ]
    )
    clf_classes = Reshape([lags, NUM_CLASSES, N_STEPS])(clf_classes)
    clf_classes = Permute([1, 3, 2])(clf_classes)
    seat = Softmax(name="seat")(clf_classes)
    return [fare, seat]


def last_timestep_mae(y_true, y_pred):
    return mean_absolute_error(y_true[:, -1], y_pred[:, -1])
last_timestep_mae.__name__ = "mae"


def causal_conv_lstm(units, res_units, kernel_size, strides, **rnn_configs):
    inputs, x = create_inputs()
    x = Conv1D(units, kernel_size, strides, padding="valid")(x)

    for _ in range(res_units):
        prev = x = LayerNormalization()(x)

        layers = [
            LSTM(units, **rnn_configs, return_sequences=True),
            LayerNormalization(),
            Activation("tanh"),
            LSTM(units, **rnn_configs, return_sequences=True),
            LayerNormalization()
        ]

        x = reduce(lambda x, func: func(x), layers, x)
        x = add([x, prev])
        x = Activation("tanh")(x)
    
    x = Dropout(0.3)(x)
    condensed_lags = (LAGS - kernel_size) // strides + 1
    outputs = _causal_output(x, condensed_lags) # NOTE: changes needed
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 1.5, "seat": 0.5},
        metrics={"fare": last_timestep_mae}
    )
    model.summary()
    return model


def tensor_board(model_comment, sub_comment=""):
    TIMESTAMP = time.strftime("%m%d_%H%M")
    log_path = f"/home/nien/tf/log/{model_comment}/{TIMESTAMP}"
    if sub_comment != "":
        log_path = "_".join([log_path, sub_comment])
    return TensorBoard(log_dir=log_path, histogram_freq=1)


def early_stopping(patience=30):
    return EarlyStopping(
        patience=patience, monitor="val_fare_mae", mode="min", restore_best_weights=True
    )


def conv_lstm_rev(units, res_units, **rnn_configs):
    inputs, x = create_inputs()
    x = Conv1D(units, 14, 1, padding="valid")(x)

    for _ in range(res_units):
        prev = x = LayerNormalization()(x)

        layers = [
            LSTM(units, **rnn_configs, return_sequences=True),
            LayerNormalization(),
            Activation("tanh"),
            LSTM(units, **rnn_configs, return_sequences=True),
            LayerNormalization()
        ]

        x = reduce(lambda x, func: func(x), layers, x)
        x = add([x, prev])
        x = Activation("tanh")(x)

    x = Flatten()(x)
    x = Dropout(0.3)(x)

    fare = Dense(N_STEPS, name="fare")(x)

    cls_days = [
        Dense(NUM_CLASSES, activation="softmax", name=f"day_{i}")(x)
        for i in range(N_STEPS)
    ]
    seat = Reshape((N_STEPS, NUM_CLASSES), name="seat")(concatenate(cls_days))

    model = Model(inputs=inputs, outputs=[fare, seat])
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 1.5, "seat": 0.5},
        metrics={"fare": last_timestep_mae}
    )
    model.summary()
    return model


def dilated_conv(filters, n_blocks, layers_per_block):
    inputs, x = create_inputs()
    for rate in [2 ** i for i in range(layers_per_block)] * n_blocks:
        x = Conv1D(
            filters, kernel_size=2, padding="causal", dilation_rate=rate, activation="relu"
        )(x)
    
    x = Dropout(0.4)(x)

    fare = Conv1D(N_STEPS, kernel_size=1, name="fare")(x)

    classes = concatenate(
        [
            Conv1D(N_STEPS, kernel_size=1, name=f"class_{i}")(x)
            for i in range(NUM_CLASSES)
        ]
    )
    classes = Reshape([LAGS, NUM_CLASSES, N_STEPS])(classes)
    classes = Permute([1, 3, 2])(classes)
    seat = Softmax(name="seat")(classes)

    model = Model(inputs=inputs, outputs=[fare, seat])
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 1.5, "seat": 0.5},
        metrics={"fare": last_timestep_mae}
    )
    model.summary()
    return model
