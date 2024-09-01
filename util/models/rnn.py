from functools import reduce

from keras.layers import (
    LSTM,
    Activation,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    LayerNormalization,
    RepeatVector,
    Reshape,
    TimeDistributed,
    add,
    concatenate,
    Permute,
    Softmax,
)
from keras.models import Model
import tensorflow as tf

from ..snapshot import LAGS, N_STEPS
from .helper import (
    NUM_CLASSES,
    create_causal_outputs,
    create_inputs,
    last_timestep_categorical_mae,
    last_timestep_mae,
    categorical_mae,
    output_names,
)


def lstm_encoder_decoder(units, n_layers=4, residual=False, **kwargs):
    inputs, x = create_inputs()

    layers_per_block = n_layers // 2
    rnn_configs = {"units": units, "kernel_regularizer": "l2"} | kwargs

    x = LSTM(**rnn_configs, dropout=0.3, return_sequences=True)(x)
    # encode
    for _ in range(layers_per_block):
        _x = LayerNormalization()(x)
        _x = LSTM(**rnn_configs, dropout=0.3, return_sequences=True)(_x)
        if residual:
            x = add([x, _x])
        else:
            x = _x
    # latent representation
    x = LayerNormalization()(x)
    x = LSTM(**rnn_configs)(x)
    x = RepeatVector(N_STEPS)(x)
    # decode
    for _ in range(layers_per_block):
        _x = LayerNormalization()(x)
        _x = LSTM(**rnn_configs, return_sequences=True)(_x)
        if residual:
            x = add([x, _x])
        else:
            x = _x

    # Regression output
    fare = TimeDistributed(
        Dense(1, activation="linear"), name=output_names[0]
    )(x)
    # Classification output
    seat = TimeDistributed(
        Dense(NUM_CLASSES, activation="softmax"), name=output_names[1]
    )(x)

    model = Model(inputs=inputs, outputs=[fare, seat])

    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 1.8, "seat": 0.2},
        metrics={"fare": "mae", "seat": categorical_mae},
    )
    model.summary()
    return model


def conv_lstm(units, n_layers, kernel_size, strides):
    inputs, x = create_inputs()
    x = Conv1D(units, kernel_size, strides)(x)

    for _ in range(n_layers):
        x = LayerNormalization()(x)
        x = LSTM(units, return_sequences=True)(x)
        x = Dropout(0.3)(x)

    fare = TimeDistributed(Dense(N_STEPS), name="fare")(x)

    classes = tf.stack(
        [
            TimeDistributed(Dense(N_STEPS), name=f"class_{i}")(x)
            for i in range(NUM_CLASSES)
        ],
        axis=-1
    )
    seat = Softmax(name="seat")(classes)

    model = Model(inputs=inputs, outputs=[fare, seat])
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 1.8, "seat": 0.2},
        metrics={"fare": last_timestep_mae, "seat": last_timestep_categorical_mae},
    )
    model.summary()
    return model


def causal_conv_lstm(units, res_units, kernel_size, strides, **rnn_configs):
    """
    Conv1D to condense the input at dim of lags. Using each lag to do next n_steps
    prediction.
    """
    inputs, x = create_inputs()
    x = Conv1D(units, kernel_size, strides, padding="valid")(x)

    for _ in range(res_units):
        prev = x = LayerNormalization()(x)

        layers = [
            LSTM(units, **rnn_configs, return_sequences=True),
            LayerNormalization(),
            Activation("tanh"),
            LSTM(units, **rnn_configs, return_sequences=True),
            LayerNormalization(),
        ]

        x = reduce(lambda x, func: func(x), layers, x)
        x = add([x, prev])
        x = Activation("tanh")(x)

    x = Dropout(0.3)(x)
    condensed_lags = (LAGS - kernel_size) // strides + 1
    outputs = create_causal_outputs(x, condensed_lags)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 1.5, "seat": 0.5},
        metrics={"fare": last_timestep_mae},
    )
    model.summary()
    return model


def window_conv_lstm(units, res_units, **rnn_configs):
    inputs, x = create_inputs()
    x = Conv1D(units, 14, 1, padding="valid")(x)

    for _ in range(res_units):
        prev = x = LayerNormalization()(x)

        layers = [
            LSTM(units, **rnn_configs, return_sequences=True),
            LayerNormalization(),
            Activation("tanh"),
            LSTM(units, **rnn_configs, return_sequences=True),
            LayerNormalization(),
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
        metrics={"fare": last_timestep_mae},
    )
    model.summary()
    return model
