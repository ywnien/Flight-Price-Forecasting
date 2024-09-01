from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    MaxPooling1D,
    ReLU,
    Reshape,
    Softmax,
    add,
    concatenate,
)
from keras.models import Model

from .helper import N_STEPS, NUM_CLASSES, create_inputs, early_stopping, tensor_board


def res_unit(units=64, kernel_size=3, downscale=False):
    def call_res_unit(x):
        if downscale:
            prev = Conv1D(units, 1, 2, "same")(x)
            prev = BatchNormalization()(prev)
            x = Conv1D(units, kernel_size, 2, "same")(x)
        else:
            prev = x
            x = Conv1D(units, kernel_size, 1, "same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(units, kernel_size, 1, "same")(x)
        x = BatchNormalization()(x)
        x = add([prev, x])
        x = ReLU()(x)
        return x

    return call_res_unit


def resnet(init_filters=64, n_blocks=(2, 2)):
    inputs, x = create_inputs()
    x = Conv1D(init_filters, 7, 2, "same")(x)
    x = MaxPooling1D(3, 2, "same")(x)
    downscale = False
    for level, n_layers in enumerate(n_blocks):
        filters = init_filters * (level + 1)
        for _ in range(n_layers):
            x = res_unit(filters, downscale=downscale)(x)
        downscale = True
    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation="relu")(x)

    fare = Dense(N_STEPS, name="fare")(x)

    classes = concatenate(
        [Dense(NUM_CLASSES, name=f"day_{i}")(x) for i in range(N_STEPS)]
    )
    classes = Reshape([N_STEPS, NUM_CLASSES])(classes)
    seat = Softmax(name="seat")(classes)
    
    model = Model(inputs, [fare, seat])
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 5, "seat": 0.5},
        metrics={"fare": "mae"},
    )
    model.summary()
    return model
