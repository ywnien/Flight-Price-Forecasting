import time

from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.layers import (Dense, Embedding, Input, Permute, Reshape, Softmax,
                          TimeDistributed, concatenate)
from keras.metrics import mean_absolute_error
import tensorflow as tf

from ..snapshot import LAGS, N_STEPS

NUM_CLASSES = 11
embed_dim = 2
input_names = ["encoded", "fare_basis"]
output_names = ["fare", "seat"]


def create_inputs():
    encoded = Input(shape=(LAGS, 34), name=input_names[0])
    fare_basis = Input(shape=(LAGS, 1), name=input_names[1])

    embedded = Embedding(input_dim=300, output_dim=embed_dim)(fare_basis)
    embedded = Reshape((LAGS, embed_dim))(embedded)
    # x = Concatenate(axis=-1)([encoded, embedded])
    # return [encoded, fare_basis], x
    return [encoded, fare_basis], concatenate([encoded, embedded])


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


def last_timestep_mae(y_true, y_pred):
    return mean_absolute_error(y_true[:, -1], y_pred[:, -1])
last_timestep_mae.__name__ = "mae"


def create_causal_outputs(x, lags=LAGS):
    fare = TimeDistributed(Dense(N_STEPS), name=output_names[0])(x)

    clf_classes = concatenate(
        [
            TimeDistributed(Dense(N_STEPS), name=f"class_{i}")(x)
            for i in range(NUM_CLASSES)
        ]
    )
    clf_classes = Reshape([lags, NUM_CLASSES, N_STEPS])(clf_classes)
    clf_classes = Permute([1, 3, 2])(clf_classes)
    seat = Softmax(name=output_names[1])(clf_classes)
    return [fare, seat]


def categorical_mae(y_true, y_pred):
    return mean_absolute_error(y_true, tf.argmax(y_pred, axis=-1))
categorical_mae.__name__ = "mae"


def last_timestep_categorical_mae(y_true, y_pred):
    return mean_absolute_error(y_true[:, -1], tf.argmax(y_pred[:, -1], axis=-1))
last_timestep_categorical_mae.__name__ = "mae"


class BriefOuput(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch+1} - ", end="", flush=True)
        self.epoch_start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        tokens = [
            f"{time.time() - self.epoch_start_time:.0f}s",
            f"fare_mae: {logs['fare_mae']:.4f}",
            f"val_fare_mae: {logs['val_fare_mae']:.4f}",
            f"seat_mae: {logs['seat_mae']:.4f}",
            f"val_seat_mae: {logs['val_seat_mae']:.4f}",

        ]
        print(*tokens, sep=" - ")
