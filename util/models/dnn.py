from keras.layers import BatchNormalization, Dense, Flatten, Reshape, Softmax, Dropout
from keras.models import Model

from .helper import N_STEPS, NUM_CLASSES, categorical_mae, create_inputs


def fully_connected(units=32, n_layers=3):
    inputs, x = create_inputs()

    x = Flatten()(x)
    x = BatchNormalization()(x)
    for _ in range(n_layers):
        x = Dense(units * N_STEPS, kernel_regularizer="l2", activation="relu")(x)
        x = Dropout(0.3)(x)

    _fare = Dense(1 * N_STEPS)(x)
    fare = Reshape([N_STEPS, 1], name="fare")(_fare)

    _seat = Dense(NUM_CLASSES * N_STEPS)(x)
    _seat = Reshape([N_STEPS, NUM_CLASSES])(_seat)
    seat = Softmax(name="seat")(_seat)

    model = Model(inputs=inputs, outputs=[fare, seat])
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": 1.8, "seat": 0.5},
        metrics={"fare": "mae", "seat": categorical_mae},
    )
    model.summary()
    return model
