from datetime import datetime

# import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.callbacks import CSVLogger, EarlyStopping
from keras.layers import (LSTM, BatchNormalization, Dense,
                          Dropout, Input, LayerNormalization, RepeatVector,
                          TimeDistributed, Layer, LSTMCell, RNN)
from keras.models import Sequential

from tstide.dataframe import valid_folds
# from tstide.forecast._keras import backtesting_keras
from tstide.transformer import TimeSeriesTransformer3D

TIMESTAMP = datetime.now().strftime("%m%d_%H%M%S")
df = pd.read_parquet("full_df.parquet")

input_timesteps=14
input_features=7
output_timesteps=3
output_features=2
# units=128
units=32


# class LNLSTMCell(LSTMCell):
#     """
#     LayerNormalization when cell return output and states.
#     VERY SLOW
#     """
#     def __init__(self, units, activation="tanh", **kwargs):
#         super().__init__(units, activation, **kwargs)
#         self.__layer_norm = LayerNormalization()
    
#     def call(self, inputs, states):
#         outputs, new_states = super().call(inputs, states)
#         norm_outputs = self.activation(self.__layer_norm(outputs))
#         norm_states = self.activation(self.__layer_norm(new_states[1]))
#         return norm_outputs, [norm_outputs, norm_states]


model = Sequential([
    Input(shape=(input_timesteps, input_features)),
    BatchNormalization(),

    # LSTM(units, return_sequences=True, kernel_regularizer="l2"),
    # Dropout(0.2),

    # LayerNormalization(),
    # LSTM(units, return_sequences=True, kernel_regularizer="l2"),
    # Dropout(0.2),

    LayerNormalization(),
    LSTM(units, kernel_regularizer="l2"),
    Dropout(0.2),

    RepeatVector(output_timesteps),

    LayerNormalization(),
    LSTM(units, return_sequences=True, kernel_regularizer="l2"),
    Dropout(0.2),

    # RNN(LNLSTMCell(units), return_sequences=True),
    # LayerNormalization(),
    # LSTM(units, return_sequences=True, kernel_regularizer="l2"),
    # Dropout(0.2),

    LayerNormalization(),
    LSTM(units, return_sequences=True),
    TimeDistributed(Dense(output_features)),
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

gen_valids = valid_folds(df, df.index.get_level_values(1), n_folds=1)
fold = next(gen_valids)

ts_trans = TimeSeriesTransformer3D(
    14,
    "legId",
    ["totalFare", "seatsRemaining"],
    y_steps=3,
)

X_train, y_train, trailing_data = ts_trans.train_test_split(fold)
X_valid, y_valid = ts_trans.train_X_y(trailing_data)

# csv_logger = CSVLogger(f"{TIMESTAMP}.csv")
early_stopping = EarlyStopping(patience=30, monitor="val_mae", mode="min", restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    # callbacks=[early_stopping, csv_logger],
    callbacks=[early_stopping],
    shuffle=False
)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.savefig(f"log_{TIMESTAMP}.png", dpi=300)

# model.save(f"LSTM_{TIMESTAMP}")

print(min(history.history["val_loss"]))
