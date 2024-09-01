from keras import activations
from keras.activations import sigmoid
from keras.layers import (
    Conv1D,
    Layer,
    Permute,
    ReLU,
    Reshape,
    Softmax,
    add,
    concatenate,
)
from keras.models import Model
import tensorflow as tf

from ..snapshot import LAGS, N_STEPS
from .helper import (
    NUM_CLASSES,
    create_inputs,
    last_timestep_categorical_mae,
    last_timestep_mae,
)


@tf.keras.utils.register_keras_serializable()
class WaveNetResidualBlock(Layer):
    def __init__(self, filters, dilation_rate, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.params = {
            "filters": filters,
            "dilation_rate": dilation_rate,
            "activation": activation
        }
        self.filter = Conv1D(filters, 2, padding="causal", dilation_rate=dilation_rate)
        self.gate = Conv1D(filters, 2, padding="causal", dilation_rate=dilation_rate)
        self.activation = activations.get(activation)
        self.conv = Conv1D(filters, 1)

    def call(self, inputs):
        z = self.activation(self.filter(inputs)) * sigmoid(self.gate(inputs))
        z = self.conv(z)
        return add([inputs, z]), z

    def get_config(self):
        return super().get_config() | self.params


def wavenet(filters, n_blocks, layers_per_block):
    inputs, x = create_inputs()

    z = Conv1D(filters, 2, padding="causal")(x)
    skip_to_last = []
    for _ in range(n_blocks):
        for i in range(layers_per_block):
            dilation_rate = 2 ** i
            z, skip = WaveNetResidualBlock(filters, dilation_rate)(z)
            skip_to_last.append(skip)
    z = ReLU()(add(skip_to_last))
    z = Conv1D(filters, 1)(z)
    z = ReLU()(z)

    fare = Conv1D(N_STEPS, 1, name="fare")(z)

    classes = concatenate(
        [
            Conv1D(N_STEPS, 1, name=f"class_{i}")(z)
            for i in range(NUM_CLASSES)
        ]
    )
    classes = Reshape([LAGS, NUM_CLASSES, N_STEPS])(classes)
    classes = Permute([1, 3, 2])(classes)
    seat = Softmax(name="seat")(classes)

    model = Model(inputs=[inputs], outputs=[fare, seat])
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 1.8, "seat": 0.2},
        metrics={"fare": last_timestep_mae, "seat": last_timestep_categorical_mae},
    )
    model.summary()
    return model


@tf.keras.utils.register_keras_serializable()
class ConditionalWaveNetResidualBlock(Layer):
    def __init__(self, filters, dilation_rate, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.params = {
            "filters": filters,
            "dilation_rate": dilation_rate,
            "activation": activation
        }
        self.filter = Conv1D(filters, 2, padding="causal", dilation_rate=dilation_rate)
        self.gate = Conv1D(filters, 2, padding="causal", dilation_rate=dilation_rate)
        self.activation = activations.get(activation)
        self.cond_filter = Conv1D(filters, 1)
        self.cond_gate = Conv1D(filters, 1)
        self.conv = Conv1D(filters, 1)

    def call(self, inputs):
        x, y = inputs

        cond_filter = self.cond_filter(y)
        cond_gate = self.cond_gate(y)
        y = self.activation(cond_filter) * sigmoid(cond_gate)

        filter = self.filter(x) + cond_filter
        gate = self.gate(x) + cond_gate
        z = self.activation(filter) * sigmoid(gate)
        z = self.conv(z)
        return add([x, z]), z, y

    def get_config(self):
        return super().get_config() | self.params


def cond_wavenet(filters, n_blocks, layers_per_block, activation="tanh"):
    inputs, x = create_inputs()
    x, h = tf.split(x, [2, x.shape[-1]-2], axis=-1)

    z = Conv1D(filters, kernel_size=2, padding="causal")(x)
    skipped = []
    for dilation_rate in [2**i for i in range(layers_per_block)] * n_blocks:
        z, skip, h = ConditionalWaveNetResidualBlock(filters, dilation_rate, activation)([z, h])
        skipped.append(skip)
    z = ReLU()(add(skipped))
    z = Conv1D(filters, kernel_size=1)(z)
    z = ReLU()(z)

    fare = Conv1D(N_STEPS, kernel_size=1, name="fare")(z)

    # classes = concatenate(
    #     [
    #         Conv1D(N_STEPS, kernel_size=1, name=f"class_{i}")(z)
    #         for i in range(NUM_CLASSES)
    #     ]
    # )
    # classes = Reshape([LAGS, NUM_CLASSES, N_STEPS])(classes)
    # classes = Permute([1, 3, 2])(classes)
    classes = [
        Conv1D(N_STEPS, kernel_size=1, name=f"class_{i}")(z)
        for i in range(NUM_CLASSES)
    ]
    classes = tf.stack(classes, axis=-1)
    
    seat = Softmax(name="seat")(classes)

    model = Model(inputs=[inputs], outputs=[fare, seat])
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 1.8, "seat": 0.2},
        metrics={"fare": last_timestep_mae, "seat": last_timestep_categorical_mae},
    )
    model.summary()
    return model
